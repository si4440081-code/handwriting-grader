"""시연 웹페이지(web/index.html)가 부르는 주소 (노트북 5-2와 같은 요청과 응답)

- GET  /v1/config, /v1/subjects, /v1/samples       화면을 처음 그릴 때
- POST /v1/grade/images                            예시 답안지(sample_ids) 또는 올린 사진(sheets)으로 채점 → 작업 번호
- GET  /v1/jobs/{job_id}                           진행률, 끝나면 결과
- POST /v1/runs/{run_id}/decisions                 선생님이 확인 문항을 정답이나 오답으로 정함
- GET  /files/{path}                               답안지, 자른 칸, 빨간펜 채점 이미지와 PDF
"""
import json
import os
import uuid
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from demo.server import Context, NotFound
from demo.subjects import trained_model, uploaded_model
from grader.model import GraderModel
from grader.tables import ANSWER_KEY, ROSTER, STUDENT_ANSWERS, TableError, classify, parse_answer_key, parse_roster, read_tables
from vision.layout import ANSWER_BOXES
from vision.pages import expand

from .schemas import PolicyIn

router = APIRouter()


def demo_of(request):
    return request.app.state.demo


@router.get('/', response_class=HTMLResponse, include_in_schema=False)
def page(request: Request):
    path = demo_of(request).settings.web_page
    if path.exists():
        return path.read_text(encoding='utf-8')
    return '<p>시연 웹페이지(web/index.html)가 없습니다. 따로 올린 페이지 주소 뒤에 ?api=이 서버 주소 를 붙이세요.</p>'


@router.get('/v1/config')
def config(request: Request):
    return demo_of(request).config()


@router.get('/v1/subjects')
def subjects(request: Request):
    return demo_of(request).subject_list()


@router.get('/v1/samples')
def samples(request: Request, subject: str = ''):
    return demo_of(request).sample_list(subject)


@router.get('/files/{path:path}', include_in_schema=False)
def files(request: Request, path: str):
    try:
        return FileResponse(demo_of(request).file_path(path))
    except NotFound:
        raise HTTPException(404) from None


def _read_upload(f, budget):
    """올린 파일 → (이름, 내용). budget은 남은 바이트 수 [list로 감싸 여러 파일에 걸쳐 줄임]"""
    data = f.file.read(budget[0] + 1)
    if len(data) > budget[0]:
        raise HTTPException(413, '올린 파일이 너무 큽니다')
    budget[0] -= len(data)
    name = os.path.basename((f.filename or '').replace('\\', '/'))
    return (name if name.strip('.') else 'sheet.jpg'), data


def _context(demo, subject, table_files):
    """올린 정답표, 명렬표, grader.json으로 채점 기준 바꾸기 → (Context 또는 None, 채점 전에 할 일 또는 None).
    정답표만 올리면 그 정답표로 신경망을 다시 학습하고(몇 초), grader.json을 같이 올리면 그 신경망으로 채점한다"""
    found, warnings, notebook_model = {}, [], None
    for name, data in table_files:
        if name.lower().endswith('.json'):
            try:
                notebook_model = GraderModel.from_dict(json.loads(data))
            except (ValueError, UnicodeDecodeError) as e:
                raise TableError(f'{name}: {e}') from None
            continue
        f, skipped = classify(read_tables(name, data))
        warnings += skipped
        for k, t in f.items():
            found.setdefault(k, t)
    base = demo.context(subject)
    if not (found or notebook_model):
        return base, None
    answer_key = parse_answer_key(found[ANSWER_KEY]) if ANSWER_KEY in found else base.answer_key
    roster = parse_roster(found[ROSTER]) if ROSTER in found else base.roster
    if STUDENT_ANSWERS in found:
        warnings.append('사진으로 채점할 때는 올린 학생 답안 표를 쓰지 않습니다')
    if not any(q in ANSWER_BOXES for q in answer_key):
        raise TableError(f'정답표에 시험지 칸({min(ANSWER_BOXES)}~{max(ANSWER_BOXES)}번)에 해당하는 문항이 없습니다')
    name = f'{subject} (올린 정답표)' if ANSWER_KEY in found else subject
    if notebook_model is not None:
        model, info, warn = uploaded_model(notebook_model, answer_key)
        return Context(name, answer_key, roster, model, info, warnings + warn), None
    if ANSWER_KEY not in found:
        return Context(name, answer_key, roster, base.model, base.model_info, warnings), None

    def prepare(progress):
        progress(0, 0, '올린 정답표로 신경망을 학습하는 중')
        model, info = trained_model(answer_key, demo.settings)
        return Context(name, answer_key, roster, model, info, warnings)
    return None, prepare


def _add_warnings(ctx, warnings):
    ctx.warnings = ctx.warnings + warnings
    return ctx


@router.post('/v1/grade/images')
def grade_images(request: Request,
                 sheets: list[UploadFile] = File([], description='답안지 사진(jpg, png, heic), PDF, ZIP'),
                 tables: list[UploadFile] = File([], description='바꿀 정답표, 명렬표(CSV, 엑셀)와 노트북에서 내보낸 grader.json'),
                 sample_ids: str = Form('', description='예시 답안지 번호를 쉼표로 (있으면 sheets는 보지 않음)'),
                 access_code: str = Form(''), t_high: float = Form(0.8, ge=0, le=1), t_low: float = Form(0.2, ge=0, le=1),
                 mixed_reads_to_review: bool = Form(True), near_miss_to_review: bool = Form(True),
                 background: bool = Form(True, description='false면 채점이 끝날 때까지 기다렸다가 결과를 바로 돌려줌'),
                 subject: str = Form('')):
    demo = demo_of(request)
    if t_low >= t_high:
        raise HTTPException(422, 't_low는 t_high보다 작아야 합니다')
    policy = PolicyIn(t_high=t_high, t_low=t_low, mixed_reads_to_review=mixed_reads_to_review,
                      near_miss_to_review=near_miss_to_review).to_policy()
    subject = subject or next(iter(demo.subjects), '')
    if subject not in demo.subjects:
        raise HTTPException(404, f'없는 과목입니다: {subject}')

    if sample_ids.strip():
        ids = [i.strip() for i in sample_ids.split(',') if i.strip()]
        if any(i not in demo.samples for i in ids):
            raise HTTPException(404, '없는 예시 답안지가 있습니다')
        subs = {demo.samples[i][0] for i in ids}
        if len(subs) > 1:
            raise HTTPException(422, '한 번에 한 과목의 답안지만 채점할 수 있습니다')
        subject = subs.pop()
        job_id, future = demo.submit([(i, demo.samples[i][1]) for i in ids], demo.context(subject), policy, uploaded=False)
    else:
        if demo.settings.access_code and access_code != demo.settings.access_code:
            raise HTTPException(403, '시연용 접근 코드가 맞지 않습니다')
        budget = [demo.settings.max_upload_mb * 1024 * 1024]
        try:
            ctx, prepare = _context(demo, subject, [_read_upload(f, budget) for f in tables])
        except TableError as e:
            raise HTTPException(422, str(e)) from None
        tag = uuid.uuid4().hex[:6]
        up_dir = demo.settings.work_dir / 'uploads' / tag
        up_dir.mkdir(parents=True)
        paths = []
        for i, f in enumerate(sheets):
            name, data = _read_upload(f, budget)
            path = up_dir / name
            if path.exists():   # 같은 이름의 파일을 여러 개 올린 경우
                path = up_dir / f'{os.path.splitext(name)[0]}_{i + 1}{os.path.splitext(name)[1]}'
            path.write_bytes(data)
            paths.append(str(path))
        try:
            pages, skipped = expand(paths, str(demo.settings.work_dir / 'pages' / f'up{tag}'), str(up_dir))
        except (ValueError, OSError) as e:
            raise HTTPException(422, f'답안지 파일을 펼치지 못했습니다: {e}') from None
        if not pages:
            raise HTTPException(422, '답안지 사진(jpg, png, heic), PDF, ZIP 중 하나를 올려 주세요')
        if not demo.reserve(len(pages)):
            raise HTTPException(429, f'오늘 새로 읽을 수 있는 답안지 수({demo.settings.daily_limit}장)를 넘었습니다. 예시 답안지로 해 보세요')
        pairs = [(f'up{tag}_{sid}', path) for sid, path in pages]   # 예시 답안지와 이름이 겹쳐도 섞이지 않게
        if ctx is not None:
            ctx.warnings = ctx.warnings + skipped
        elif skipped:
            train = prepare
            prepare = lambda progress: _add_warnings(train(progress), skipped)
        job_id, future = demo.submit(pairs, ctx, policy, uploaded=True, prepare=prepare)

    if not background:
        future.result()
        job = demo.job(job_id)
        if job['status'] == 'error':
            raise HTTPException(500, job['error'])
        return job['result']
    return {'job_id': job_id, 'status': 'running'}


@router.get('/v1/jobs/{job_id}')
def job(request: Request, job_id: str):
    try:
        return demo_of(request).job(job_id)
    except NotFound:
        raise HTTPException(404, '채점 작업을 찾을 수 없습니다 (서버가 다시 시작됐을 수 있어요)') from None


class TeacherDecision(BaseModel):
    sheet_id: str
    question: int
    decision: Literal['correct', 'wrong', 'undo']


@router.post('/v1/runs/{run_id}/decisions')
def decide(request: Request, run_id: str, d: TeacherDecision):
    """선생님이 확인 문항을 정답이나 오답으로 정함 → 성적표와 빨간펜 시험지를 다시 계산해서 돌려줌"""
    try:
        return demo_of(request).decide(run_id, d.sheet_id, d.question, d.decision)
    except NotFound as e:
        raise HTTPException(404, str(e)) from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
