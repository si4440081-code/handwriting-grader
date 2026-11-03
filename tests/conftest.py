import csv
import json
import os
import re
import threading
from collections import defaultdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from demo.settings import Settings
from vision.reader import SheetReader

ROOT = Path(__file__).resolve().parent.parent
TRUTH = ROOT / 'tests' / 'fixtures' / '정답_실제.csv'   # 예시 답안지마다 학생이 실제로 쓴 답 (멘티 자료, 베이스라인 표용)


class TruthCall:
    """Gemini 대신 학생이 실제로 쓴 답을 돌려주는 가짜 읽기. 사진 정렬, 칸 자르기, 판정은 진짜로 돈다.
    overrides: {(시험지, k번째 읽기): {문항: 글자}}로 특정 읽기만 다르게 돌려줄 수 있다"""

    def __init__(self, overrides=None):
        self.sheets = defaultdict(dict)
        self.by_stem = defaultdict(set)
        for r in csv.DictReader(open(TRUTH, encoding='utf-8-sig')):
            s = self.sheets[(r['과목'], r['시험지'])]
            s.update({'class': r['반'], 'number': r['번호'], 'name': r['이름']})
            s.setdefault('answers', {})[r['문항']] = {'text': r['실제답'], 'messy': False}
            self.by_stem[r['시험지']].add(r['과목'])
        self.overrides = overrides or {}
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, path, prompt, timeout):
        sid = os.path.basename(path)[:-len('_sheet.png')]
        prefix, stem = sid.split('_', 1)
        if (prefix, stem) not in self.sheets and stem not in self.by_stem:   # 올린 PDF 쪽(_p01), 겹친 이름(_2)
            stem = re.sub(r'(_p\d+)?(_\d+)?$', '', stem)
        subject = prefix if (prefix, stem) in self.sheets else next(iter(self.by_stem[stem]))
        with self.lock:
            k = sum(c == sid for c in self.calls)
            self.calls.append(sid)
        out = json.loads(json.dumps(self.sheets[(subject, stem)]))
        for q, text in self.overrides.get((sid, k), {}).items():
            out['answers'][str(q)]['text'] = text
        return json.dumps(out, ensure_ascii=False), 100


def make_settings(work, **kw):
    kw.setdefault('read_workers', 2)
    kw.setdefault('prefetch', False)
    return Settings(data_dir=ROOT / 'data', work_dir=Path(work), **kw)


def make_client(work, call=None, **kw):
    settings = make_settings(work, **kw)
    reader = SheetReader(call, str(Path(work) / 'gemini_cache.json'), 'fake-vision' if call else None)
    return TestClient(create_app(settings, reader))


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    """텍스트 채점 주소용 (AI 읽기 없음)"""
    with make_client(tmp_path_factory.mktemp('work')) as c:
        yield c
