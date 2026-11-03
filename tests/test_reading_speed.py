"""읽기 속도: 'safe' 방식의 처음 두 번을 동시에 보내고, 서버를 켤 때 예시 답안지를 여러 장 동시에 미리 읽는다"""
import time
from pathlib import Path

from conftest import ROOT, TruthCall, make_settings
from demo.server import DemoServer
from vision.layout import crop_sheet
from vision.reader import ReadStats, SheetReader

DELAY = 0.6   # Gemini 호출 한 번이 걸리는 시간 대신


class SlowTruth(TruthCall):
    def __call__(self, path, prompt, timeout):
        time.sleep(DELAY)
        return super().__call__(path, prompt, timeout)


def test_first_two_reads_are_sent_together(tmp_path):
    from demo.subjects import load_subject
    settings = make_settings(tmp_path)
    subject = load_subject(str(ROOT / 'data' / 'subjects' / '역사'), settings)
    sid, _, path = next(x for x in subject.samples if x[0] == '역사_김민준')
    call = SlowTruth()
    reader = SheetReader(call, str(tmp_path / 'cache.json'), 'fake')
    crops = crop_sheet(path, str(tmp_path / 'sheets' / sid))
    t0 = time.time()
    ident, per_q, _ = reader.read_sheet(sid, crops, subject.questions, subject.answer_key, ReadStats(), 'safe', 90)
    spent = time.time() - t0
    assert len(call.calls) == 2 and all(len(r) == 2 for r in per_q.values())   # 두 번 읽어 같아서 끝
    assert spent < DELAY * 1.6, f'{spent:.2f}초: 차례로 보냈다면 {DELAY * 2:.1f}초 이상'


def test_prefetch_reads_every_sample_in_parallel(tmp_path):
    call = SlowTruth()
    settings = make_settings(tmp_path, prefetch=True, read_workers=4)
    server = DemoServer(settings, SheetReader(call, str(tmp_path / 'cache.json'), 'fake'))
    n = sum(len(s.samples) for s in server.subjects.values())
    t0 = time.time()
    server.prefetch().join(timeout=120)
    spent = time.time() - t0
    assert len(server.reader.cache) == 2 * n                                    # 38장 모두 두 번씩 읽어 저장
    assert spent < n * DELAY * 2 / 3, f'{spent:.1f}초: 한 장씩 차례로 읽었다면 {n * DELAY * 2:.0f}초'
