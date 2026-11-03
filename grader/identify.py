"""학생 식별: 반+번호로 명렬표에서 찾고 이름으로 검증한다 (노트북 4-5, 프로젝트 정리 문서 3절)"""
import re
from collections import Counter

MATCHED, REVIEW = 'matched', 'review'   # 자동 매칭, 교사 확인


def digits(s):
    """손글씨에서 읽은 반, 번호에서 숫자만 ("3번" → "3", 숫자가 없으면 "")"""
    m = re.search(r'\d+', str(s or ''))
    return m.group() if m else ''


def identify(sheets, roster):
    """답안지마다 (상태, 사유). roster: {(반, 번호): 이름}, 없으면 None (읽은 반, 번호를 그대로 씀)"""
    keys = [(s.class_no, s.number, s.name) for s in sheets]
    counts = Counter(keys)
    out = []
    for c, n, name in keys:
        if not (c.isdigit() and n.isdigit()):
            out.append((REVIEW, '반 또는 번호를 읽지 못함'))
        elif roster is None:
            out.append((MATCHED, '명렬표 없음: 읽은 반, 번호 사용'))
        elif (int(c), int(n)) not in roster:
            out.append((REVIEW, f'명렬표에 없는 번호 ({c}반 {n}번)'))
        elif roster[(int(c), int(n))] != name:
            out.append((REVIEW, f'번호와 이름이 다름 (명렬표: {roster[(int(c), int(n))]}, 읽은 이름: {name or "없음"})'))
        elif counts[(c, n, name)] > 1:
            out.append((REVIEW, f'같은 학생 답안지가 {counts[(c, n, name)]}장'))
        else:
            out.append((MATCHED, '반, 번호, 이름 일치'))
    return out


def missing_students(sheets, statuses, roster):
    """자동 매칭된 답안지가 없는 학생과, 이름이 같아 그 학생 것일 수 있는 교사 확인 답안지"""
    if roster is None:
        return []
    matched = {(int(s.class_no), int(s.number)) for s, (status, _) in zip(sheets, statuses) if status == MATCHED}
    unmatched = [s for s, (status, _) in zip(sheets, statuses) if status != MATCHED]
    return [{'class_no': c, 'number': n, 'name': name,
             'candidate_sheets': [s.sheet_id for s in unmatched if s.name == name]}
            for (c, n), name in sorted(roster.items()) if (c, n) not in matched]
