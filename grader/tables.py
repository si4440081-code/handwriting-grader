"""정답표, 명렬표, 학생 답안 표 읽기 (노트북 1-3, 1-4)

CSV, TSV, TXT, 엑셀(시트 여러 개 포함) 어느 것이든 받고, 열 이름을 보고 어떤 표인지 알아서 구분한다.
- `정답` 열이 있으면 정답표, `1번`, `Q1`, `문항1` 같은 열이 있으면 학생 답안, `반, 번호, 이름`이 있으면 명렬표
- 열 이름이 조금 달라도 알아듣는다 (문제번호 → 문항, 점수 → 배점, 성명 → 이름 등)
"""
import csv
import io
import os
import re
from dataclasses import dataclass

from .answer_key import Question
from .sheets import Sheet, parse_cell

TABLE_EXT = ('.csv', '.tsv', '.txt', '.xlsx', '.xls')
ALIASES = {   # 열 이름이 조금 달라도 알아듣기
    '문항': ['문항', '문항번호', '문제', '문제번호', 'question', 'q'],
    '배점': ['배점', '점수', '배점(점)', 'points'],
    '정답': ['정답', '답', '모범답안', 'answer'],
    '허용답안': ['허용답안', '허용', '인정답안', '동의어', 'allowed'],
    '등록오답': ['등록오답', '오답', '오답예시', 'wrong'],
    '반': ['반', '학급', 'class'],
    '번호': ['번호', '출석번호', 'number', 'no'],
    '이름': ['이름', '성명', '학생', 'name'],
    '시험지': ['시험지', '시험지번호', '답안지', 'sheet', 'sheet_id'],
}
ANSWER_KEY, ROSTER, STUDENT_ANSWERS = '정답표', '명렬표', '학생 답안'


class TableError(ValueError):
    pass


@dataclass
class Table:
    name: str
    columns: list[str]
    rows: list[dict[str, str]]

    @property
    def kind(self):
        """열을 보고 정답표 / 학생 답안 / 명렬표를 구분 (모르면 None)"""
        if '정답' in self.columns:
            return ANSWER_KEY
        if any(re.fullmatch(r'\d+번', c) for c in self.columns):
            return STUDENT_ANSWERS
        if {'반', '번호', '이름'} <= set(self.columns):
            return ROSTER
        return None


def decode(raw, label):
    """엑셀은 CSV를 UTF-8(BOM 포함)이나 CP949로 저장하므로 둘 다 받는다"""
    for encoding in ('utf-8-sig', 'cp949'):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise TableError(f'{label}: 글자 인코딩을 읽을 수 없습니다 (UTF-8 또는 CP949로 저장하세요)')


def _value(v):
    s = '' if v is None else str(v).strip()
    if s in ('nan', 'None', 'NaN'):
        return ''
    return re.sub(r'^(\d+)\.0$', r'\1', s)   # 엑셀 숫자 20.0 → 20


def _clean(name, header, body):
    """열 이름 정리(별칭, 문항 열) + 값 정리 + 완전히 빈 행 제거 → Table"""
    columns = [_value(c) for c in header]
    rename = {}
    for canon, names in ALIASES.items():
        if canon not in columns:
            hit = next((c for c in columns if c.lower().replace(' ', '') in names), None)
            if hit:
                rename[hit] = canon
    for c in columns:   # 학생 답안 문항 열: 1, Q1, 문항1, 1번 → '1번'
        m = re.fullmatch(r'(?:문항|q)?\s*(\d+)\s*번?', c.lower())
        if m and c not in rename and c not in ALIASES:
            rename[c] = f'{int(m.group(1))}번'
    columns = [rename.get(c, c) for c in columns]
    rows = []
    for values in body:
        row = {c: _value(v) for c, v in zip(columns, values) if c}
        if any(row.values()):
            rows.append({c: row.get(c, '') for c in columns if c})
    return Table(name, [c for c in columns if c], rows)


def read_tables(name, data):
    """파일 이름과 내용(bytes) → [Table]. 엑셀은 시트마다 표 하나"""
    ext = os.path.splitext(name)[1].lower()
    if ext == '.xls':
        raise TableError(f'{name}: 옛날 엑셀(.xls)은 못 읽습니다. 엑셀에서 "다른 이름으로 저장" → .xlsx나 CSV로 저장해 주세요')
    if ext == '.xlsx':
        import openpyxl
        try:
            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        except Exception as e:
            raise TableError(f'{name}: 엑셀 파일을 열 수 없습니다 ({type(e).__name__})') from None
        tables = []
        for ws in wb.worksheets:
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            if rows:
                tables.append(_clean(f'{name}:{ws.title}', rows[0], rows[1:]))
        return tables
    if ext not in TABLE_EXT:
        raise TableError(f'{name}: 표 파일이 아닙니다 (CSV, TSV, 엑셀만 받습니다)')
    text = decode(data, name)
    sep = '\t' if ext == '.tsv' or text.count('\t') > text.count(',') else ','
    rows = list(csv.reader(io.StringIO(text), delimiter=sep))
    return [_clean(name, rows[0], rows[1:])] if rows else []


def read_table_file(path):
    with open(path, 'rb') as f:
        return read_tables(os.path.basename(path), f.read())


def classify(tables):
    """표 여러 개 → {종류: Table}, 건너뛴 표 설명. 같은 종류가 또 나오면 처음 것만 쓴다"""
    found, skipped = {}, []
    for t in tables:
        k = t.kind
        if k is None:
            skipped.append(f'어떤 표인지 알 수 없어 건너뜀: {t.name} (열: {t.columns})')
        elif k in found:
            skipped.append(f'{k}가 이미 있어 건너뜀: {t.name}')
        else:
            found[k] = t
    return found, skipped


def _split(s):
    return tuple(a.strip() for a in str(s).split('|') if a.strip())


def _check(errors, label):
    if errors:
        raise TableError(f'{label}에 문제가 있습니다: ' + '; '.join(errors))


def parse_answer_key(table):
    """정답표 → {문항: Question}. 열이 빠졌거나, 문항, 배점이 숫자가 아니거나, 정답이 비었거나, 문항이 겹치면 몇 행인지 알려 준다"""
    _check([f"'{c}' 열이 없습니다" for c in ('문항', '배점', '정답') if c not in table.columns], '정답표')
    errors, key = [], {}
    for i, r in enumerate(table.rows):
        line = i + 2
        if not r['문항'].isdigit():
            errors.append(f"{line}행: 문항 번호 '{r['문항']}'이(가) 숫자가 아닙니다")
        if not r['배점'].isdigit() or int(r['배점']) <= 0:
            errors.append(f"{line}행: 배점 '{r['배점']}'이(가) 양의 정수가 아닙니다")
        if not r['정답']:
            errors.append(f'{line}행: 정답이 비어 있습니다')
        if r['문항'].isdigit() and int(r['문항']) in key:
            errors.append(f"{line}행: 문항 번호가 겹칩니다 ({r['문항']})")
        if not errors:
            q = int(r['문항'])
            key[q] = Question(q, r['정답'], int(r['배점']), _split(r.get('허용답안', '')), _split(r.get('등록오답', '')),
                              r.get('메모', ''))
    _check(errors, '정답표')
    if not key:
        raise TableError('정답표에 문항이 없습니다')
    return key


def parse_roster(table):
    """명렬표 → {(반, 번호): 이름}"""
    errors, roster = [], {}
    for i, r in enumerate(table.rows):
        if not (r['반'].isdigit() and r['번호'].isdigit()):
            errors.append(f"{i + 2}행: 반, 번호가 숫자가 아님 ({r['반']}반 {r['번호']}번)")
            continue
        k = (int(r['반']), int(r['번호']))
        if k in roster:
            errors.append(f'{i + 2}행: {k[0]}반 {k[1]}번이 두 번 나옵니다')
        roster[k] = r['이름']
    _check(errors, '명렬표')
    return roster


def parse_sheets(table):
    """학생 답안 표 → [Sheet]. 칸 표시는 sheets.parse_cell 참고 (* 지저분함, ! 칸 밖, | 여러 번 읽음)"""
    questions = {c: int(c[:-1]) for c in table.columns if re.fullmatch(r'\d+번', c)}
    sheets = [Sheet(r.get('시험지') or f'sheet{i + 1:02d}', r.get('반', ''), r.get('번호', ''), r.get('이름', ''),
                    {q: parse_cell(r[c]) for c, q in questions.items()})
              for i, r in enumerate(table.rows)]
    ids = [s.sheet_id for s in sheets]
    if len(set(ids)) != len(ids):
        raise TableError('학생 답안 표에 시험지 이름이 겹칩니다')
    return sheets


def load_one(name, data, kind):
    """파일 하나에서 원하는 종류의 표를 찾아 해석한다 (예: 엑셀 한 파일에 정답표와 명렬표 시트)"""
    found, _ = classify(read_tables(name, data))
    if kind not in found:
        raise TableError(f'{name}에서 {kind}를 찾지 못했습니다 '
                         f'(정답표는 문항, 배점, 정답 열, 명렬표는 반, 번호, 이름 열, 학생 답안은 1번, 2번 같은 열이 필요합니다)')
    return {ANSWER_KEY: parse_answer_key, ROSTER: parse_roster, STUDENT_ANSWERS: parse_sheets}[kind](found[kind])
