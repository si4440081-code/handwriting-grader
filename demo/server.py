"""시연 서버의 채점 과정 (노트북 5-2의 api_grade, render_run, 선생님 판정, 작업 진행률)

사진 → 정렬과 칸 자르기 → AI 읽기 → 학생 식별과 판정 → 빨간펜 채점 시험지 → 웹페이지가 쓰는 결과.
채점은 한 번에 하나씩 차례로 돌린다(AI 사용량 제한 보호). 채점 결과는 메모리에 두므로 서버를 다시 켜면 사라진다.
"""
import datetime
import logging
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from PIL import Image

from grader.judge import REVIEW
from grader.identify import MATCHED
from grader.pipeline import judge, report
from grader.sheets import Cell, Sheet
from vision.layout import ANSWER_BOXES, crop_sheet
from vision.reader import GeminiCall, NoReading, ReadFailed, ReaderUnavailable, ReadStats, SheetReader
from vision.redpen import mark_sheet

from .subjects import load_subjects

log = logging.getLogger('demo')


class NotFound(Exception):
    pass


@dataclass
class Context:
    """채점에 쓰는 정답표, 명렬표, 신경망 (과목 그대로이거나, 올린 정답표, 명렬표, grader.json으로 바꾼 것)"""
    subject: str
    answer_key: dict
    roster: dict | None
    model: object
    model_info: dict
    warnings: list = field(default_factory=list)


@dataclass
class Run:
    run_id: str
    ctx: Context
    pairs: list                     # [(시험지, 답안지 이미지 경로)]
    crops: dict                     # 시험지 → SheetCrops
    judged: tuple                   # pipeline.judge 결과 (답안지, 식별, 판정, 문항, 경고)
    reading: dict
    decisions: dict = field(default_factory=dict)   # (시험지, 문항) → 'correct' | 'wrong'
    images: dict = field(default_factory=dict)      # 시험지 → 빨간펜 이미지 주소
    version: int = 0


def make_reader(settings):
    call = GeminiCall(settings.gemini_api_key, settings.gemini_model) if settings.gemini_api_key else None
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    return SheetReader(call, str(settings.work_dir / 'gemini_cache.json'), settings.gemini_model,
                       extra_cache=str(settings.data_dir / 'gemini_cache.json'))


class DemoServer:
    def __init__(self, settings, reader=None):
        self.settings = settings
        for d in ('pages', 'sheets', 'runs', 'uploads'):
            (settings.work_dir / d).mkdir(parents=True, exist_ok=True)
        self.reader = reader or make_reader(settings)
        self.subjects = load_subjects(settings)
        self.samples = {sid: (name, path, title) for name, s in self.subjects.items() for sid, title, path in s.samples}
        self.jobs, self.runs = {}, {}
        self.usage = {'date': '', 'sheets': 0}
        self.lock = threading.RLock()
        self.queue = ThreadPoolExecutor(max_workers=1)   # 채점은 한 번에 하나씩

    # ── 웹페이지가 처음에 부르는 것 ─────────────────────────────
    def used_today(self):
        with self.lock:
            today = datetime.date.today().isoformat()
            if self.usage['date'] != today:
                self.usage.update(date=today, sheets=0)
            return self.usage['sheets']

    def config(self):
        # 웹페이지는 "AI 읽기 모델: {vision_model}"로 보여 주므로, 키나 결제 문제로 멈춘 동안은 모델 이름 뒤에 사유를 붙인다
        problem = self.reader.problem
        model = self.reader.model_name if self.reader.live else None
        return {'live_reading': self.reader.live, 'vision_model': f'{model} (지금 사용 불가: {problem})' if problem else model,
                'reading_problem': problem,
                'daily_limit': self.settings.daily_limit, 'used_today': self.used_today(),
                'access_code_required': bool(self.settings.access_code)}

    def subject_list(self):
        return [{'name': n, 'questions': len(s.answer_key), 'samples': len(s.samples)} for n, s in self.subjects.items()]

    def sample_list(self, subject=''):
        return [{'id': sid, 'subject': n, 'title': title, 'image_url': self.url(path)}
                for n, s in self.subjects.items() if not subject or n == subject for sid, title, path in s.samples]

    def url(self, path):
        rel = os.path.relpath(path, self.settings.work_dir)
        return '/files/' + rel.replace(os.sep, '/')

    def file_path(self, rel):
        """/files/ 뒤의 경로 → 실제 파일. 펼친 답안지, 자른 칸, 채점 결과 폴더 밖은 열지 않는다"""
        root = self.settings.work_dir.resolve()
        full = (root / rel).resolve()
        if not any(full.is_relative_to(root / d) for d in ('pages', 'sheets', 'runs')) or not full.is_file():
            raise NotFound(rel)
        return full

    def context(self, name):
        s = self.subjects[name]
        return Context(name, s.answer_key, s.roster, s.model, s.model_info)

    # ── 사용량 ──────────────────────────────────────────────────
    def reserve(self, n):
        """새로 읽을 답안지 n장을 하루 상한 안에서 잡아 둔다. 넘으면 False"""
        with self.lock:
            if self.used_today() + n > self.settings.daily_limit:
                return False
            self.usage['sheets'] += n
            return True

    def release(self, n):
        with self.lock:
            self.usage['sheets'] = max(0, self.usage['sheets'] - n)

    # ── 채점 작업 ───────────────────────────────────────────────
    def submit(self, pairs, ctx, policy, uploaded, prepare=None):
        """채점을 차례에 넣고 작업 번호를 돌려준다. prepare(progress)는 채점 전에 할 일(올린 정답표로 학습 등)"""
        job_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.jobs[job_id] = {'status': 'running', 'progress': {'done': 0, 'total': len(pairs), 'message': '차례를 기다리는 중'}}

        def progress(done, total, message):
            with self.lock:
                self.jobs[job_id]['progress'] = {'done': done, 'total': total, 'message': message}

        def work():
            try:
                c = prepare(progress) if prepare else ctx
                result = self.grade(pairs, c, policy, uploaded, progress)
                with self.lock:
                    self.jobs[job_id].update(status='done', result=result)
            except Exception as e:
                log.error('채점 실패: %s', traceback.format_exc())
                if uploaded:
                    self.release(len(pairs))
                with self.lock:
                    self.jobs[job_id].update(status='error', error=f'{type(e).__name__}: {e}')

        future = self.queue.submit(work)
        return job_id, future

    def job(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise NotFound(job_id)
            return dict(self.jobs[job_id])

    def read_sheets(self, pairs, ctx, uploaded, progress, stats):
        """답안지마다 칸을 자르고 AI로 읽는다 → ([Sheet], {시험지: SheetCrops}, 경고)"""
        s = self.settings
        questions = [q for q in ctx.answer_key if q in ANSWER_BOXES]
        strategy, limit = (s.upload_read_strategy, s.upload_time_limit) if uploaded else (s.sample_read_strategy, s.sample_time_limit)
        crops = {sid: crop_sheet(path, str(s.work_dir / 'sheets' / sid)) for sid, path in pairs}
        progress(0, len(pairs), 'AI가 읽는 중')
        sheets, warnings = {}, []
        with ThreadPoolExecutor(max_workers=max(1, s.read_workers)) as ex:
            jobs = {sid: ex.submit(self.reader.read_sheet, sid, crops[sid], questions, ctx.answer_key, stats, strategy, limit,
                                   self.cache_aliases(sid))
                    for sid, _ in pairs}
            for k, (sid, _) in enumerate(pairs, 1):
                c = crops[sid]
                try:
                    (cls, num, name), per_q, messy = jobs[sid].result()
                    answers = {q: Cell(per_q[q], messy[q], c.outside[q], image=c.cells[q]) for q in questions}
                except (TimeoutError, NoReading, RuntimeError) as e:
                    if isinstance(e, TimeoutError):
                        note = f'AI가 {limit}초 안에 읽지 못함 → 교사가 직접 확인'
                    elif isinstance(e, NoReading):
                        note = 'AI 읽기 키가 없어 읽지 못함 → 교사가 직접 확인'
                    elif isinstance(e, ReadFailed):
                        note = f'AI 읽기 실패: {e.reason} → 교사가 직접 확인'
                    else:
                        note = 'AI 호출 실패 → 교사가 직접 확인'
                    if isinstance(e, ReaderUnavailable):   # 모든 답안지에 같은 사유라 경고는 한 번만
                        w = f'AI 읽기를 쓸 수 없습니다: {e.reason}. 키와 결제 상태를 확인하세요 ({e.detail[:120]})'
                        if w not in warnings:
                            warnings.append(w)
                    elif not isinstance(e, NoReading):
                        warnings.append(f'{sid}: AI가 읽지 못함 ({self.reader.errors.get(sid, e)})')
                    cls = num = name = ''
                    answers = {q: Cell([''], False, c.outside[q], note=note, image=c.cells[q]) for q in questions}
                sheets[sid] = Sheet(sid, cls, num, name, answers)
                progress(k, len(pairs), f'{sid} 읽음')
        return [sheets[sid] for sid, _ in pairs], crops, warnings

    def cache_aliases(self, sid):
        """노트북에서 미리 읽어 둔 결과(gemini_cache.json)를 쓰기 위한 다른 이름.
        노트북은 1~4단계 과목의 답안지를 과목 이름 없이 저장하므로 기본 과목의 예시 답안지만 맞춰 본다"""
        if sid in self.samples and self.samples[sid][0] == self.settings.default_subject:
            return [self.samples[sid][2]]
        return []

    def grade(self, pairs, ctx, policy, uploaded, progress=lambda *a: None):
        stats = ReadStats()
        sheets, crops, read_warnings = self.read_sheets(pairs, ctx, uploaded, progress, stats)
        judged = judge(ctx.answer_key, ctx.roster, sheets, ctx.model, policy)
        skipped = [q for q in ctx.answer_key if q not in ANSWER_BOXES]
        warnings = ctx.warnings + read_warnings + judged[4]
        if skipped:
            warnings.append(f'시험지에 칸이 없는 문항은 채점하지 않았습니다: {skipped}')
        run = Run(uuid.uuid4().hex[:12], ctx, pairs, crops, judged[:4] + (warnings,),
                  stats.to_dict(self.reader.model_name if self.reader.live else None))
        os.makedirs(self.settings.work_dir / 'runs' / run.run_id)
        progress(len(pairs), len(pairs), '빨간펜으로 채점하는 중')
        result = self.render(run, [sid for sid, _ in pairs])   # 아직 다른 요청이 볼 수 없는 결과라 잠금 밖에서 그림
        with self.lock:
            self.runs[run.run_id] = run
        return result

    # ── 선생님 판정 ─────────────────────────────────────────────
    def decide(self, run_id, sheet_id, question, decision):
        """선생님이 확인 문항을 정답이나 오답으로 정함('undo'는 되돌리기) → 성적표와 빨간펜 시험지를 다시 계산"""
        with self.lock:
            if run_id not in self.runs:
                raise NotFound('채점 결과를 찾을 수 없습니다 (서버가 다시 시작됐을 수 있어요)')
            run = self.runs[run_id]
            rows = run.judged[2]
            row = next((r for r in rows if r['sheet_id'] == sheet_id and r['question'] == question), None)
            if row is None or row['decision'] != REVIEW:
                raise ValueError('선생님 확인 문항이 아닙니다')
            if decision not in ('correct', 'wrong', 'undo'):
                raise ValueError('decision은 correct, wrong, undo 중 하나여야 합니다')
            if decision == 'undo':
                run.decisions.pop((sheet_id, question), None)
            else:
                run.decisions[(sheet_id, question)] = decision
            return self.render(run, [sheet_id])

    # ── 결과 만들기 ─────────────────────────────────────────────
    def render(self, run, remark):
        """채점 결과 → 성적표 계산, remark에 든 시험지의 빨간펜 그림 다시 그리기 → 웹페이지가 쓰는 결과"""
        sheets, statuses, rows, questions, warnings = run.judged
        rep = report(run.ctx.answer_key, run.ctx.roster, sheets, statuses, rows, questions, warnings, run.decisions)
        run_dir = self.settings.work_dir / 'runs' / run.run_id
        scores = {s['sheet_id']: s for s in rep['scores']}
        if remark:
            run.version += 1
            for sid in remark:
                finals = {a['question']: a['final'] for a in rep['answers'] if a['sheet_id'] == sid}
                sc = scores[sid]
                out = run_dir / f'graded_{sid}_v{run.version}.png'
                mark_sheet(sid, run.crops[sid].flat, finals, sc['score'], sc['review_count'],
                           sc['identification'] == MATCHED, out)
                run.images[sid] = self.url(out)
            pages = [Image.open(self.file_path(run.images[sid][len('/files/'):])) for sid, _ in run.pairs]
            pages[0].save(run_dir / 'graded.pdf', save_all=True, append_images=pages[1:])
        image_of = {(r['sheet_id'], r['question']): r['image'] for r in rows}
        for item in rep['review_items']:
            item['image_url'] = self.url(image_of[(item['sheet_id'], item['question'])])
        keep = ('sheet_id', 'student', 'question', 'read_answer', 'reads', 'messy', 'outside', 'p_correct',
                'decision', 'final', 'reason')
        return {
            'run_id': run.run_id, 'subject': run.ctx.subject, 'can_decide': True,
            'summary': rep['summary'],
            'answers': [{k: a[k] for k in keep} for a in rep['answers']],
            'review_items': rep['review_items'],
            'scores': rep['scores'],
            'missing_students': rep['missing_students'],
            'sheet_images': [{'sheet_id': sid, 'aligned': run.crops[sid].aligned,
                              'outside_questions': [q for q in questions if run.crops[sid].outside.get(q)],
                              'graded_url': run.images[sid]} for sid, _ in run.pairs],
            'pdf_url': f'/files/runs/{run.run_id}/graded.pdf?v={run.version}',
            'reading': run.reading,
            'model_info': {k: run.ctx.model_info.get(k) for k in ('source', 'val_accuracy', 'epochs')},
            'warnings': rep['warnings'],
        }

    # ── 서버를 켤 때 예시 답안지 미리 읽기 ─────────────────────
    def prefetch(self):
        """예시 답안지를 미리 읽어 둠 → 웹페이지에서 누르면 바로 채점됨 (키가 있을 때만, 따로 도는 스레드에서)"""
        if not (self.settings.prefetch and self.reader.live):
            return None

        def read_one(s, sid, path):
            crops = crop_sheet(path, str(self.settings.work_dir / 'sheets' / sid))
            self.reader.read_sheet(sid, crops, s.questions, s.answer_key, ReadStats(), self.settings.sample_read_strategy,
                                   self.settings.sample_time_limit, self.cache_aliases(sid))

        def work():
            t0 = time.time()
            for name, s in self.subjects.items():   # 기본 과목부터, 과목 안에서는 READ_WORKERS장씩 동시에
                ok = 0
                with ThreadPoolExecutor(max_workers=max(1, self.settings.read_workers)) as ex:
                    futures = {ex.submit(read_one, s, sid, path): sid for sid, _, path in s.samples}
                    for f in as_completed(futures):
                        try:
                            f.result()
                            ok += 1
                        except ReaderUnavailable as e:   # 키, 결제 문제는 남은 답안지에도 똑같이 걸리므로 미리 읽기를 멈춤
                            ex.shutdown(wait=False, cancel_futures=True)
                            log.warning('미리 읽기를 멈춤: %s (%s). 문제를 해결한 뒤 서버를 다시 켜면 다시 미리 읽습니다',
                                        e.reason, e.detail[:120])
                            return
                        except Exception as e:
                            log.warning('%s 미리 읽기 실패: %s', futures[f], e)
                log.info('%s: 예시 답안지 %d/%d장 미리 읽음 (시작부터 %.0f초)', name, ok, len(s.samples), time.time() - t0)

        t = threading.Thread(target=work, daemon=True, name='prefetch')
        t.start()
        return t
