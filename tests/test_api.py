"""텍스트 채점 주소 (/health, /v1/grade, /v1/grade/csv)"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / 'samples'


def sample_files(**override):
    files = {k: (f'{k}.csv', (SAMPLES / f'{k}.csv').read_bytes(), 'text/csv')
             for k in ('answer_key', 'roster', 'student_answers')}
    files.update({k: (name, v, 'application/octet-stream') for k, (name, v) in override.items()})
    return files


def test_health_reports_model(client):
    body = client.get('/health').json()
    assert body['status'] == 'ok' and body['model']['answer_key'] == 'e96414021891'


def test_grade_csv_with_samples(client):
    r = client.post('/v1/grade/csv', files=sample_files())
    assert r.status_code == 200
    summary = r.json()['summary']
    assert (summary['correct'], summary['wrong'], summary['review']) == (87, 37, 31)


def test_grade_csv_with_probability_only_policy(client):
    r = client.post('/v1/grade/csv', files=sample_files(),
                    data={'mixed_reads_to_review': 'false', 'near_miss_to_review': 'false'})
    summary = r.json()['summary']
    assert (summary['correct'], summary['wrong'], summary['review']) == (89, 40, 26)


def test_grade_csv_accepts_cp949_and_excel(client):
    """한국어 윈도우 엑셀은 CSV를 CP949로 저장하고, 정답표와 명렬표를 엑셀 한 파일에 시트로 넣기도 한다"""
    cp949 = (SAMPLES / 'answer_key.csv').read_text(encoding='utf-8-sig').encode('cp949')
    assert client.post('/v1/grade/csv', files=sample_files(answer_key=('a.csv', cp949))).status_code == 200
    xlsx = (ROOT / 'data' / 'subjects' / '역사' / '역사_시험자료.xlsx').read_bytes()
    r = client.post('/v1/grade/csv', files=sample_files(answer_key=('역사.xlsx', xlsx), roster=('역사.xlsx', xlsx)))
    assert r.status_code == 200 and r.json()['summary']['correct'] == 87


def test_grade_csv_rejects_missing_columns(client):
    r = client.post('/v1/grade/csv', files=sample_files(answer_key=('a.csv', '문항,정답\n1,이순신\n'.encode())))
    assert r.status_code == 422 and '찾지 못했습니다' not in r.json()['detail'] and '배점' in r.json()['detail']


def test_grade_json_accepts_cell_objects_and_marker_strings(client):
    body = {
        'answer_key': [{'number': 1, 'answer': '이순신', 'allowed': ['충무공 이순신'], 'registered_wrong': ['원균']},
                       {'number': 2, 'answer': 'DNA', 'registered_wrong': ['RNA']}],
        'roster': [{'class_no': 1, 'number': 1, 'name': '김민준'}],
        'sheets': [
            {'sheet_id': 's1', 'class_no': 1, 'number': '1번', 'name': '김민준',
             'answers': {'1': {'reads': ['이순신'] * 2}, '2': {'reads': ['dna', 'dna']}}},
            {'sheet_id': 's2', 'class_no': '', 'number': '', 'name': '이서연', 'answers': {'1': '!이순신', '2': 'RNA'}},
        ],
    }
    r = client.post('/v1/grade', json=body)
    assert r.status_code == 200
    answers = {(a['sheet_id'], a['question']): a for a in r.json()['answers']}
    assert answers[('s1', 1)]['decision'] == 'correct' and answers[('s1', 2)]['decision'] == 'correct'   # 대소문자 무시
    assert answers[('s2', 1)]['decision'] == 'review' and answers[('s2', 1)]['outside']
    assert answers[('s2', 2)]['decision'] == 'wrong'
    scores = {s['sheet_id']: s for s in r.json()['scores']}
    assert scores['s1']['identification'] == 'matched'                    # '1번'에서 숫자만 읽음
    assert scores['s2']['identification_reason'] == '반 또는 번호를 읽지 못함'


@pytest.mark.parametrize('policy', [{'t_high': 0.3, 't_low': 0.5}, {'t_high': 1.5}, {'mixed_reads': 'sometimes'}])
def test_grade_json_rejects_bad_policy(client, policy):
    body = {'answer_key': [{'number': 1, 'answer': '이순신'}], 'roster': [], 'sheets': [], 'policy': policy}
    assert client.post('/v1/grade', json=body).status_code == 422


def test_grade_json_rejects_duplicate_sheet_ids(client):
    sheet = {'sheet_id': 's1', 'class_no': 1, 'number': 1, 'name': '김민준', 'answers': {'1': '이순신'}}
    body = {'answer_key': [{'number': 1, 'answer': '이순신'}], 'roster': [], 'sheets': [sheet, sheet]}
    r = client.post('/v1/grade', json=body)
    assert r.status_code == 422 and 's1' in r.json()['detail']
