"""채점 전체 흐름: 학생 식별 → 답안마다 특징, P(정답), 판정 → 교사 확인 목록과 성적표 (노트북 4-5, 4-6)

judge()가 판정까지 하고, report()가 선생님 판정을 반영해 성적표를 만든다. 둘을 나눈 이유는 선생님이 확인 문항을
정답이나 오답으로 정할 때마다 판정은 그대로 두고 성적표만 다시 계산하기 위해서다.
"""
from collections import Counter
from dataclasses import replace

from .features import FEATURES, make_features, top_read
from .identify import MATCHED, digits, identify, missing_students
from .judge import CORRECT, REVIEW, WRONG, Policy, decide, near_miss
from .sheets import Cell


def judge(answer_key, roster, sheets, model, policy=Policy()):
    """→ (식별 정리한 답안지, 식별 결과, 답안별 판정, 채점한 문항, 경고)"""
    duplicated = [k for k, v in Counter(s.sheet_id for s in sheets).items() if v > 1]
    if duplicated:
        raise ValueError(f'시험지 번호가 겹칩니다: {duplicated}')
    sheets = [replace(s, class_no=digits(s.class_no), number=digits(s.number), name=str(s.name or '').strip())
              for s in sheets]

    # 답안지에 한 번이라도 나온 문항만 채점한다 (학생 답안 표에 "N번" 열이 있거나, 시험지에 칸이 있는 문항)
    asked = {q for s in sheets for q in s.answers}
    questions = [q for q in answer_key if q in asked]
    warnings = []
    if asked - set(answer_key):
        warnings.append(f'정답표에 없는 문항은 채점하지 않았습니다: {sorted(asked - set(answer_key))}')

    statuses = identify(sheets, roster)
    rows = []
    for sheet in sheets:
        for q in questions:
            question = answer_key[q]
            cell = sheet.answers.get(q, Cell(['']))
            rows.append({
                'sheet_id': sheet.sheet_id, 'student': sheet.student, 'question': q,
                'read_answer': top_read(cell.reads)[0] or '(빈칸)', 'reads': cell.reads,
                'messy': cell.messy, 'outside': cell.outside, 'image': cell.image, 'note': cell.note,
                'features': make_features(cell.reads, question, cell.messy),
                'near_miss': near_miss(cell.reads, question),
            })

    # 노트북처럼 P(정답)을 소수 셋째 자리로 반올림한 뒤 판정한다
    ps = model.predict([[r['features'][k] for k in FEATURES] for r in rows]) if rows else []
    for r, p in zip(rows, ps):
        r['p_correct'] = round(float(p), 3)
        r['decision'], r['reason'] = decide(r['p_correct'], r['features'], policy, r['outside'], r.pop('near_miss'))
        if r['note']:   # AI가 읽지 못한 칸은 무조건 교사 확인
            r['decision'], r['reason'] = REVIEW, r['note']
    return sheets, statuses, rows, questions, warnings


def report(answer_key, roster, sheets, statuses, rows, questions, warnings, decisions=None):
    """판정 결과 + 선생님 판정({(시험지, 문항): 'correct' 또는 'wrong'}) → 요약, 교사 확인 목록, 성적표"""
    decisions = decisions or {}
    answers = []
    for r in rows:
        teacher = decisions.get((r['sheet_id'], r['question'])) if r['decision'] == REVIEW else None
        final = teacher or r['decision']   # 선생님이 정한 것이 우선
        answers.append({**{k: v for k, v in r.items() if k not in ('image', 'note')}, 'final': final,
                        'points': answer_key[r['question']].points if final == CORRECT else 0})

    # 성적표: 확정 점수 + 교사 확인이 끝나면 더해질 수 있는 최대 점수
    max_score = sum(answer_key[q].points for q in questions)
    scores = []
    for sheet, (status, reason) in zip(sheets, statuses):
        mine = [a for a in answers if a['sheet_id'] == sheet.sheet_id]
        pending = [a for a in mine if a['final'] == REVIEW]
        scores.append({
            'sheet_id': sheet.sheet_id, 'student': sheet.student,
            'identification': status, 'identification_reason': reason,
            'score': sum(a['points'] for a in mine), 'max_score': max_score,
            'review_count': len(pending), 'pending_points': sum(answer_key[a['question']].points for a in pending),
            'status': 'confirmed' if status == MATCHED and not pending else 'needs_review',
        })

    c = Counter(r['decision'] for r in rows)
    return {
        'summary': {
            'sheets': len(sheets), 'answers': len(rows),
            'correct': c[CORRECT], 'wrong': c[WRONG], 'review': c[REVIEW],
            'auto_rate': round((c[CORRECT] + c[WRONG]) / len(rows), 3) if rows else 0.0,
            'identification_review': sum(status != MATCHED for status, _ in statuses),
            'confirmed_sheets': sum(s['status'] == 'confirmed' for s in scores),
            'teacher_decided': sum(a['final'] != a['decision'] for a in answers),
        },
        'answers': answers,
        'review_items': [{'sheet_id': r['sheet_id'], 'student': r['student'], 'question': r['question'],
                          'reads': r['reads'], 'reason': r['reason'],
                          'teacher_decision': decisions.get((r['sheet_id'], r['question']))}
                         for r in rows if r['decision'] == REVIEW],
        'scores': scores,
        'missing_students': missing_students(sheets, statuses, roster),
        'warnings': warnings,
    }


def grade(answer_key, roster, sheets, model, policy=Policy()):
    """answer_key: {문항: Question}, roster: {(반, 번호): 이름} 또는 None, sheets: [Sheet] → 채점 결과 dict"""
    sheets, statuses, rows, questions, warnings = judge(answer_key, roster, sheets, model, policy)
    return report(answer_key, roster, sheets, statuses, rows, questions, warnings)
