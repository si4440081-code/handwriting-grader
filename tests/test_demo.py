"""시연 서버 (웹페이지가 부르는 주소). 예시 답안지 사진을 실제로 정렬하고 자르며, AI 읽기만 TruthCall로 바꿔 끼운다"""
import io
import json
import time
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from conftest import ROOT, TruthCall, make_client

SHEETS = ROOT / 'data' / 'subjects'
# 예시_답안지_설명.txt: 과목별 01 정답만, 02 허용 표기, 03 등록오답, 04 애매한 답, 05 지저분한 칸, 06 섞인 답안 + 학생 식별 문제
ORDER = {'국어': ['김민준', '이서연', '박지호', '이정민', '이정빈', '최서아'],
         '영어': ['정해린', '강도윤', '윤하은', '장예준', '임지우', '한서준'],
         '수학': ['오유나', '서지안', '신하준', '권채원', '황민서', '오유나_2'],
         '과학': ['송다은', '전주원', '홍지민', '고은우', '문서윤', '김하늘']}
# 노트북 1-5와 같은 동작이라 멘티 의도(선생님 확인)와 다른 칸. tests/test_engine.py의 xfail 테스트 참고
KNOWN_ORDER_CASE = {('과학_고은우', 3)}   # "반응중화": 허용답안 "중화"가 가장 가까워 순서 바뀜으로 잡히지 않고 자동 오답
ID_PROBLEM = {'국어': '번호와 이름이 다름', '영어': '반 또는 번호를 읽지 못함', '수학': '같은 학생 답안지가 2장', '과학': '명렬표에 없는 번호'}


@pytest.fixture(scope='module')
def fake():
    return TruthCall(overrides={('역사_이서연', 1): {1: '이순산 장군'}})   # 두 번째 읽기만 다르게 → 세 번째까지 읽음


@pytest.fixture(scope='module')
def demo(tmp_path_factory, fake):
    with make_client(tmp_path_factory.mktemp('work'), fake) as c:
        yield c


def wait(client, response):
    assert response.status_code == 200, response.text
    body = response.json()
    for _ in range(600):
        job = client.get(f"/v1/jobs/{body['job_id']}").json()
        if job['status'] != 'running':
            assert job['status'] == 'done', job
            return job['result']
        time.sleep(0.05)
    raise AssertionError('채점이 끝나지 않음')


def grade_samples(client, ids, **form):
    return wait(client, client.post('/v1/grade/images', data={'sample_ids': ','.join(ids), **form}))


def finals(result, sid):
    return {a['question']: a['final'] for a in result['answers'] if a['sheet_id'] == sid}


def test_config_subjects_samples(demo):
    assert demo.get('/v1/config').json() == {'live_reading': True, 'vision_model': 'fake-vision', 'reading_problem': None,
                                             'daily_limit': 40, 'used_today': 0, 'access_code_required': False}
    subjects = demo.get('/v1/subjects').json()
    assert subjects[0]['name'] == '역사' and {s['name'] for s in subjects} == {'역사', '국어', '영어', '수학', '과학'}
    samples = demo.get('/v1/samples', params={'subject': '영어'}).json()
    assert {s['title'] for s in samples} == set(ORDER['영어'])
    assert demo.get(samples[0]['image_url']).status_code == 200
    assert '시연' in demo.get('/').text


@pytest.mark.parametrize('path', ['../api/main.py', 'pages/../../api/main.py', 'gemini_cache.json', 'uploads/x.jpg'])
def test_files_stay_inside_work_folders(demo, path):
    assert demo.get(f'/files/{path}').status_code == 404


@pytest.mark.parametrize('subject', ORDER)
def test_sample_sheets_match_mentee_intent(demo, subject):
    """예시_답안지_설명.txt에 적힌 의도대로 판정된다"""
    ids = [f'{subject}_{n}' for n in ORDER[subject]]
    r = grade_samples(demo, ids)
    s01, s02, s03, s04, s05, s06 = ids
    assert set(finals(r, s01).values()) == {'correct'}                   # 01 정답만 → 모두 동그라미
    assert set(finals(r, s02).values()) == {'correct'}                   # 02 허용 표기, 띄어쓰기, 대소문자 → 모두 동그라미
    assert set(finals(r, s03).values()) == {'wrong'}                     # 03 등록오답 → 모두 빗금
    four = [a for a in r['answers'] if a['sheet_id'] == s04]             # 04 애매한 답 → 선생님 확인, 빈칸은 오답
    assert all(a['final'] == ('wrong' if a['read_answer'] == '(빈칸)' else 'review') for a in four
               if (a['sheet_id'], a['question']) not in KNOWN_ORDER_CASE)
    five = [a for a in r['answers'] if a['sheet_id'] == s05]             # 05 답 두 개, 칸 밖 글씨 → 선생님 확인
    assert any('두 개' in a['reason'] for a in five)
    assert all(a['final'] == 'review' for a in five if a['outside'] or ',' in a['read_answer'])
    reasons = {s['sheet_id']: s['identification_reason'] for s in r['scores'] if s['identification'] != 'matched'}
    assert any(ID_PROBLEM[subject] in v for k, v in reasons.items() if k == s06)   # 06 학생 식별 문제
    assert r['summary']['sheets'] == 6 and len(r['sheet_images']) == 6
    assert all(x['aligned'] for x in r['sheet_images'])


def test_result_has_every_field_the_web_page_reads(demo):
    r = grade_samples(demo, ['역사_정해린', '역사_김민준'])
    assert {'run_id', 'can_decide', 'summary', 'answers', 'review_items', 'scores', 'missing_students', 'sheet_images',
            'pdf_url', 'reading', 'model_info', 'warnings'} <= set(r)
    assert r['model_info']['source'] == 'trained_for_answer_key' and r['model_info']['val_accuracy'] > 0.85
    assert {'reads', 'new_calls', 'cached', 'tokens', 'avg_call_seconds', 'model'} <= set(r['reading'])
    item = r['review_items'][0]
    assert {'image_url', 'teacher_decision'} <= set(item) and demo.get(item['image_url']).status_code == 200
    jung = next(x for x in r['sheet_images'] if x['sheet_id'] == '역사_정해린')
    assert jung['outside_questions'] == [1, 2, 4]                          # 칸 밖 오른쪽에 쓴 답
    assert demo.get(jung['graded_url']).status_code == 200 and demo.get(r['pdf_url']).status_code == 200
    assert all(a['final'] == 'review' for a in r['answers'] if a['sheet_id'] == '역사_정해린' and a['question'] in (1, 2, 4))


def test_mixed_reads_go_to_review_and_cached_reads_are_reused(demo, fake):
    r = grade_samples(demo, ['역사_이서연'])
    a1 = next(a for a in r['answers'] if a['question'] == 1)
    # 처음 두 번은 동시에 보내므로 어느 쪽이 다르게 읽혔는지 순서는 정해지지 않음
    assert sorted(a1['reads']) == sorted(['이순신 장군', '이순산 장군', '이순신 장군'])
    assert '엇갈림' in a1['reason'] and a1['final'] == 'review'
    assert r['reading']['new_calls'] == 3
    again = grade_samples(demo, ['역사_이서연'])
    assert again['reading']['new_calls'] == 0 and again['reading']['cached'] == 3   # 한 번 읽은 답안지는 다시 부르지 않음


def test_teacher_decision_updates_scores_and_red_pen(demo):
    r = grade_samples(demo, ['영어_장예준'])
    item = next(i for i in r['review_items'])
    before = r['scores'][0]
    after = demo.post(f"/v1/runs/{r['run_id']}/decisions",
                      json={'sheet_id': item['sheet_id'], 'question': item['question'], 'decision': 'correct'}).json()
    assert after['scores'][0]['score'] == before['score'] + 20
    assert after['summary']['teacher_decided'] == 1
    assert next(i for i in after['review_items'] if i['question'] == item['question'])['teacher_decision'] == 'correct'
    assert after['sheet_images'][0]['graded_url'] != r['sheet_images'][0]['graded_url']   # 빨간펜 다시 그림
    undone = demo.post(f"/v1/runs/{r['run_id']}/decisions",
                       json={'sheet_id': item['sheet_id'], 'question': item['question'], 'decision': 'undo'}).json()
    assert undone['scores'][0]['score'] == before['score'] and undone['summary']['teacher_decided'] == 0


def test_teacher_decision_errors(demo):
    r = grade_samples(demo, ['영어_정해린'])
    ok = r['answers'][0]
    assert demo.post(f"/v1/runs/{r['run_id']}/decisions",
                     json={'sheet_id': ok['sheet_id'], 'question': ok['question'], 'decision': 'wrong'}).status_code == 422
    assert demo.post('/v1/runs/nope/decisions', json={'sheet_id': 'x', 'question': 1, 'decision': 'correct'}).status_code == 404
    assert demo.get('/v1/jobs/nope').status_code == 404


@pytest.mark.parametrize('form, status', [({'sample_ids': '없는답안지'}, 404), ({'sample_ids': '영어_정해린,국어_김민준'}, 422),
                                          ({'sample_ids': '영어_정해린', 't_high': '0.3', 't_low': '0.5'}, 422),
                                          ({'subject': '체육'}, 404)])
def test_grade_images_rejects_bad_requests(demo, form, status):
    assert demo.post('/v1/grade/images', data=form).status_code == status


def upload(client, files, tables=(), **form):
    multipart = [('sheets', (name, data, 'application/octet-stream')) for name, data in files]
    multipart += [('tables', (name, data, 'application/octet-stream')) for name, data in tables]
    return client.post('/v1/grade/images', files=multipart, data={'subject': '영어', **form})


def photo(subject, name):
    return (f'{name}.jpg', (SHEETS / subject / 'sheets' / f'{name}.jpg').read_bytes())


def test_upload_photo_counts_usage_and_reads_fast(demo, fake):
    used = demo.get('/v1/config').json()['used_today']
    r = wait(demo, upload(demo, [photo('영어', '강도윤'), photo('영어', '장예준')]))
    assert demo.get('/v1/config').json()['used_today'] == used + 2
    sid = next(s['sheet_id'] for s in r['scores'] if s['sheet_id'].endswith('_강도윤'))
    assert set(finals(r, sid).values()) == {'correct'}
    assert fake.calls.count(sid) == 1                                     # 애매한 칸이 없으면 한 번만 읽음 ('fast')
    other = next(s['sheet_id'] for s in r['scores'] if s['sheet_id'].endswith('_장예준'))
    assert fake.calls.count(other) == 2                                   # 애매한 칸이 있으면 한 번 더


def test_upload_zip_and_pdf(demo):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:   # 다른 폴더의 같은 이름 → 시험지 이름이 겹치지 않게
        z.writestr('a/강도윤.jpg', photo('영어', '강도윤')[1])
        z.writestr('b/강도윤.jpg', photo('영어', '강도윤')[1])
        z.writestr('__MACOSX/a/._강도윤.jpg', b'junk')
    pdf = io.BytesIO()
    page = Image.open(SHEETS / '영어' / 'sheets' / '윤하은.jpg').convert('RGB')
    page.save(pdf, format='PDF', save_all=True, append_images=[page])
    r = wait(demo, upload(demo, [('묶음.zip', buf.getvalue()), ('윤하은.pdf', pdf.getvalue())]))
    ids = [s['sheet_id'] for s in r['scores']]
    assert len(ids) == 4 and len(set(ids)) == 4
    dup = [s for s in r['scores'] if '강도윤' in s['sheet_id']]
    assert all('같은 학생 답안지가 2장' in s['identification_reason'] for s in dup)


def test_upload_answer_key_retrains_and_grader_json_is_used(demo):
    xlsx = (SHEETS / '영어' / '영어_시험자료.xlsx').read_bytes()
    r = wait(demo, upload(demo, [photo('영어', '윤하은')], tables=[('영어_시험자료.xlsx', xlsx)]))
    assert r['subject'] == '영어 (올린 정답표)' and r['model_info']['source'] == 'trained_for_answer_key'
    assert set(finals(r, r['scores'][0]['sheet_id']).values()) == {'wrong'}

    # 노트북 3-7에서 내보낸 grader.json과 같은 형식 (정답표 지문이 같으면 경고 없음)
    from demo.subjects import trained_model
    from grader.answer_key import fingerprint
    from demo.settings import Settings
    model, _ = trained_model(demo.app.state.demo.subjects['영어'].answer_key, demo.app.state.demo.settings)
    exported = model.to_dict()
    exported['info'] = {'source': 'notebook', 'answer_key': fingerprint(demo.app.state.demo.subjects['영어'].answer_key),
                        'val_accuracy': 0.9}
    r = wait(demo, upload(demo, [photo('영어', '윤하은')], tables=[('grader.json', json.dumps(exported).encode())]))
    assert r['model_info'] == {'source': 'notebook_upload', 'val_accuracy': 0.9, 'epochs': None} and not r['warnings']
    exported['info']['answer_key'] = 'aaaaaaaaaaaa'
    r = wait(demo, upload(demo, [photo('영어', '윤하은')], tables=[('grader.json', json.dumps(exported).encode())]))
    assert any('다른 정답표' in w for w in r['warnings'])


@pytest.mark.parametrize('tables, message', [([('grader.json', b'{"layers": []}')], 'grader.json'),
                                             ([('key.csv', '문항,배점,정답\n7,20,x\n'.encode())], '시험지 칸'),
                                             ([('key.csv', '문항,배점,정답\n1,이십,x\n'.encode())], '양의 정수')])
def test_upload_rejects_bad_tables(demo, tables, message):
    r = upload(demo, [photo('영어', '윤하은')], tables=tables)
    assert r.status_code == 422 and message in r.json()['detail']


def test_upload_rejects_non_sheet_files(demo):
    r = upload(demo, [('메모.txt', b'hello')])
    assert r.status_code == 422


def test_access_code_and_daily_limit(tmp_path):
    with make_client(tmp_path / 'a', TruthCall(), access_code='teamj', daily_limit=1) as c:
        assert c.get('/v1/config').json()['access_code_required'] is True
        assert upload(c, [photo('영어', '강도윤')]).status_code == 403
        assert upload(c, [photo('영어', '강도윤'), photo('영어', '윤하은')], access_code='teamj').status_code == 429
        wait(c, upload(c, [photo('영어', '강도윤')], access_code='teamj'))
        assert upload(c, [photo('영어', '윤하은')], access_code='teamj').status_code == 429
        assert c.post('/v1/grade/images', data={'sample_ids': '영어_정해린'}).status_code == 200   # 예시 답안지는 상한과 무관


def test_without_key_every_cell_goes_to_teacher(tmp_path):
    with make_client(tmp_path / 'b') as c:
        assert c.get('/v1/config').json()['live_reading'] is False
        r = c.post('/v1/grade/images', data={'sample_ids': '역사_김민준', 'background': 'false'}).json()
        assert r['summary']['review'] == 5
        assert all(i['reason'] == 'AI 읽기 키가 없어 읽지 못함 → 교사가 직접 확인' for i in r['review_items'])
