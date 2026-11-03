"""손글씨 단답형 채점 보조 API + 독립 시연 서버

    uvicorn api.main:app            # http://127.0.0.1:8000 시연 웹페이지, /docs 에서 모든 주소를 호출해 볼 수 있다

시연 웹페이지용 (api/demo.py, 노트북 5단계와 같은 요청과 응답)
- 사진으로 채점: /v1/config, /v1/subjects, /v1/samples, /v1/grade/images, /v1/jobs/{id}, /v1/runs/{id}/decisions
텍스트 채점 (이미 읽은 결과부터 받음)
- POST /v1/grade      JSON으로 정답표, 명렬표, 답안지를 보내 채점
- POST /v1/grade/csv  교사가 엑셀로 만든 표 세 개(CSV 또는 엑셀)를 그대로 올려 채점
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from demo.server import DemoServer
from demo.settings import Settings
from grader.answer_key import Question
from grader.model import DEFAULT_MODEL_PATH, GraderModel
from grader.pipeline import grade
from grader.sheets import Cell, Sheet, parse_cell
from grader.tables import ANSWER_KEY, ROSTER, STUDENT_ANSWERS, TableError, load_one

from . import demo
from .schemas import GradeRequest, GradeResponse, HealthResponse, PolicyIn

text = APIRouter()


def _run(request, answer_key, roster, sheets, policy_in):
    try:
        result = grade(answer_key, roster, sheets, request.app.state.model, policy_in.to_policy())
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {**result, 'policy': policy_in}


@text.get('/health', response_model=HealthResponse)
def health(request: Request):
    return {'status': 'ok', 'model': request.app.state.model.info}


@text.post('/v1/grade', response_model=GradeResponse)
def grade_json(body: GradeRequest, request: Request):
    numbers = [q.number for q in body.answer_key]
    if len(set(numbers)) != len(numbers):
        raise HTTPException(422, f'정답표의 문항 번호가 겹칩니다: {numbers}')
    answer_key = {q.number: Question(q.number, q.answer, q.points, tuple(q.allowed), tuple(q.registered_wrong), q.memo)
                  for q in body.answer_key}
    roster = {(s.class_no, s.number): s.name for s in body.roster}
    sheets = [Sheet(s.sheet_id, '' if s.class_no is None else str(s.class_no), '' if s.number is None else str(s.number),
                    s.name, {q: parse_cell(c) if isinstance(c, str) else Cell(c.reads, c.messy, c.outside)
                             for q, c in s.answers.items()})
              for s in body.sheets]
    return _run(request, answer_key, roster, sheets, body.policy)


@text.post('/v1/grade/csv', response_model=GradeResponse)
def grade_csv(
    request: Request,
    answer_key: UploadFile = File(description='정답표 (CSV, TSV, 엑셀)'),
    roster: UploadFile = File(description='명렬표 (CSV, TSV, 엑셀)'),
    student_answers: UploadFile = File(description='학생 답안 표 (CSV, TSV, 엑셀)'),
    t_high: float = Form(0.8, ge=0, le=1),
    t_low: float = Form(0.2, ge=0, le=1),
    mixed_reads_to_review: bool = Form(True),
    near_miss_to_review: bool = Form(True),
):
    if t_low >= t_high:
        raise HTTPException(422, 't_low는 t_high보다 작아야 합니다')
    try:
        key = load_one(answer_key.filename or 'answer_key.csv', answer_key.file.read(), ANSWER_KEY)
        students = load_one(roster.filename or 'roster.csv', roster.file.read(), ROSTER)
        sheets = load_one(student_answers.filename or 'student_answers.csv', student_answers.file.read(), STUDENT_ANSWERS)
    except TableError as e:
        raise HTTPException(422, str(e)) from None
    policy_in = PolicyIn(t_high=t_high, t_low=t_low, mixed_reads_to_review=mixed_reads_to_review,
                         near_miss_to_review=near_miss_to_review)
    return _run(request, key, students, sheets, policy_in)


def create_app(settings=None, reader=None):
    """settings, reader를 넘기면 그것으로 시연 서버를 만든다 (테스트에서 가짜 AI 읽기를 쓸 때)"""
    log = logging.getLogger('demo')   # 과목 불러오기, 미리 읽기 안내 (다른 라이브러리의 로그 수준은 건드리지 않음)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(message)s'))
        log.addHandler(handler)
    log.setLevel(logging.INFO)

    @asynccontextmanager
    async def lifespan(app):
        app.state.model = GraderModel.load(os.environ.get('GRADER_MODEL_PATH', DEFAULT_MODEL_PATH))
        app.state.demo = DemoServer(settings or Settings.from_env(), reader)
        app.state.demo.prefetch()
        yield

    app = FastAPI(title='손글씨 단답형 채점 보조 API', version='0.2.0', lifespan=lifespan,
                  description='답안지 사진을 펴고, 칸을 자르고, AI가 정답을 모른 채 읽고, 신경망이 P(정답)을 계산해 '
                              '정답, 오답, 교사 확인으로 나누고, 빨간펜 채점 시험지와 성적표를 만든다.')
    # 시연 웹페이지를 서버와 따로 올려도(Vercel, GitHub Pages 등) 부를 수 있게
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET', 'POST'], allow_headers=['*'])
    app.include_router(demo.router)
    app.include_router(text)
    return app


app = create_app()
