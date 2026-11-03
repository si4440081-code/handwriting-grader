"""AI 읽기가 거절될 때: 다시 시도해도 소용없는 오류(키, 결제, 모델)는 기다리지 않고 바로 선생님 확인으로 보낸다"""
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from conftest import TruthCall, make_settings
from vision.reader import SheetReader, classify

CREDITS = '402 RESOURCE_EXHAUSTED. {\'error\': {\'code\': 402, \'message\': \'Your prepayment credits are depleted.\'}}'


class ApiError(Exception):
    """google-genai의 APIError처럼 code 속성과 '코드 상태. 내용' 문자열을 가진 오류"""

    def __init__(self, code, text):
        super().__init__(text)
        self.code = code


class Failing:
    """처음 fail_times번은 error를 던지고, 그 뒤에는 학생이 실제로 쓴 답을 돌려준다"""

    def __init__(self, error, fail_times=10 ** 6):
        self.error, self.fail_times, self.calls, self.truth = error, fail_times, 0, TruthCall()

    def __call__(self, path, prompt, timeout):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return self.truth(path, prompt, timeout)


def client_for(work, call, block_seconds=60):
    reader = SheetReader(call, str(Path(work) / 'gemini_cache.json'), 'fake-vision', block_seconds=block_seconds)
    return TestClient(create_app(make_settings(work, read_workers=1), reader))


def grade(c, ids):
    r = c.post('/v1/grade/images', data={'sample_ids': ','.join(ids), 'background': 'false'})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize('error, reason', [
    (ApiError(402, CREDITS), '결제 크레딧이 없음'),
    (Exception(CREDITS), '결제 크레딧이 없음'),                                     # code 속성 없이 문자열만 있는 경우
    (ApiError(400, '400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key.'), 'API 키가 올바르지 않음'),
    (ApiError(403, '403 PERMISSION_DENIED.'), '권한이 없음'),
    (ApiError(404, '404 NOT_FOUND. models/gemini-x is not found'), '모델 이름을 찾을 수 없음'),
    (ApiError(429, '429 RESOURCE_EXHAUSTED. Quota exceeded ... limit: 0, model: gemini-x'), '무료 한도 0'),
])
def test_fatal_errors_are_classified(error, reason):
    found, everywhere = classify(error)
    assert reason in found and everywhere


@pytest.mark.parametrize('error', [ApiError(429, '429 RESOURCE_EXHAUSTED. Please retry in 3s'), ApiError(503, '503 UNAVAILABLE'),
                                   TimeoutError('timed out')])
def test_temporary_errors_are_retried(error):
    assert classify(error) is None


def test_credit_error_fails_fast_once_for_all_sheets(tmp_path):
    call = Failing(ApiError(402, CREDITS))
    with client_for(tmp_path, call) as c:
        t0 = time.time()
        r = grade(c, ['역사_김민준', '역사_이서연'])
        assert time.time() - t0 < 10                    # 30초, 90초를 기다리지 않음
        assert call.calls <= 2                          # 첫 장의 동시 읽기 2번 뒤에는 부르지 않음
        assert r['summary']['review'] == 10
        assert {i['reason'] for i in r['review_items']} == {'AI 읽기 실패: 결제 크레딧이 없음 → 교사가 직접 확인'}
        assert sum('AI 읽기를 쓸 수 없습니다: 결제 크레딧이 없음' in w for w in r['warnings']) == 1
        config = c.get('/v1/config').json()
        assert config['reading_problem'] == '결제 크레딧이 없음'
        assert config['vision_model'] == 'fake-vision (지금 사용 불가: 결제 크레딧이 없음)'


def test_reading_resumes_after_block_time(tmp_path):
    """멈춘 시간이 지나면(예: 그 사이 크레딧 충전) 다시 부른다"""
    call = Failing(ApiError(402, CREDITS), fail_times=2)   # 첫 장의 동시 읽기 2번이 모두 거절됨
    with client_for(tmp_path, call, block_seconds=0) as c:
        r = grade(c, ['역사_김민준'])
        assert r['summary']['review'] == 5
        assert c.get('/v1/config').json()['reading_problem'] is None
        r = grade(c, ['역사_김민준'])
        assert r['summary']['correct'] == 5            # 다시 불러서 제대로 읽음


def test_rate_limit_is_still_retried(tmp_path):
    call = Failing(ApiError(429, '429 RESOURCE_EXHAUSTED. Please retry in 0s'), fail_times=1)
    with client_for(tmp_path, call) as c:
        r = grade(c, ['역사_김민준'])
        assert r['summary']['correct'] == 5 and call.calls >= 2
