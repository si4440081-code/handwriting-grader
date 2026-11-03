"""요청과 응답 형식 (OpenAPI 문서 /docs 에 그대로 나온다)"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from grader.judge import Policy

Decision = Literal['correct', 'wrong', 'review']   # 정답, 오답, 교사 확인


# ---------- 요청 ----------

class QuestionIn(BaseModel):
    """정답표 한 줄 (answer_key.csv 한 행)"""
    number: int = Field(description='문항 번호')
    points: int = Field(20, ge=0, description='배점')
    answer: str = Field(min_length=1, description='대표 정답')
    allowed: list[str] = Field([], description='함께 인정할 답')
    registered_wrong: list[str] = Field([], description='확실한 오답')
    memo: str = ''


class StudentIn(BaseModel):
    """명렬표 한 줄"""
    class_no: int
    number: int
    name: str


class CellIn(BaseModel):
    """답안 칸 하나. 문자열로 보내면 CSV와 같은 표시(* 지저분함, ! 칸 밖, | 여러 번 읽음)로 해석한다"""
    reads: list[str] = Field(min_length=1, description='같은 칸을 여러 번 읽은 결과')
    messy: bool = Field(False, description='지운 자국, 답 두 개 등')
    outside: bool = Field(False, description='칸 밖에 쓴 글씨')


class SheetIn(BaseModel):
    """답안지 한 장. 반과 번호는 손글씨에서 읽은 값이라 비어 있거나 숫자가 아닐 수 있다"""
    sheet_id: str = Field(min_length=1)
    class_no: str | int | None = None
    number: str | int | None = None
    name: str = ''
    answers: dict[int, CellIn | str] = Field(description='문항 번호 → 답안 칸')


class PolicyIn(BaseModel):
    """판정 기준. 기본값은 노트북 1-2 설정과 같다"""
    t_high: float = Field(0.8, ge=0, le=1, description='P(정답)이 이 값 이상이면 정답')
    t_low: float = Field(0.2, ge=0, le=1, description='P(정답)이 이 값 이하이면 오답')
    mixed_reads_to_review: bool = Field(True, description='읽기 결과 중 일부만 정답이면 교사 확인 (웹페이지의 체크 상자)')
    mixed_reads: Literal['strict', 'lenient', 'off'] | None = Field(
        None, description="엇갈린 읽기 규칙을 직접 고를 때: 'strict' 하나라도 엇갈리면, 'lenient' 80% 미만이면, 'off' 확률에 맡김")
    near_miss_to_review: bool = Field(True, description='한 글자 차이이거나 정답 뒤에 말이 붙은 답은 교사 확인')

    @model_validator(mode='after')
    def check_thresholds(self):
        if self.t_low >= self.t_high:
            raise ValueError('t_low는 t_high보다 작아야 합니다')
        return self

    def to_policy(self):
        mixed = self.mixed_reads or ('strict' if self.mixed_reads_to_review else 'off')
        return Policy(self.t_high, self.t_low, mixed, self.near_miss_to_review)


class GradeRequest(BaseModel):
    answer_key: list[QuestionIn] = Field(min_length=1)
    roster: list[StudentIn]
    sheets: list[SheetIn]
    policy: PolicyIn = PolicyIn()


# ---------- 응답 ----------

class AnswerResult(BaseModel):
    sheet_id: str
    student: str
    question: int
    read_answer: str = Field(description='가장 많이 나온 읽기 결과 (띄어쓰기와 기호 정리 후)')
    reads: list[str]
    messy: bool
    outside: bool
    features: dict[str, float] = Field(description='신경망 입력 특징 8개 (노트북 1-3 이름 그대로)')
    p_correct: float = Field(description='P(정답), 소수 셋째 자리')
    decision: Decision = Field(description='AI 판정')
    final: Decision = Field(description='선생님 판정을 반영한 최종 판정')
    reason: str
    points: int = Field(description='득점 (교사 확인은 확인 전까지 0점)')


class ReviewItem(BaseModel):
    sheet_id: str
    student: str
    question: int
    reads: list[str]
    reason: str
    teacher_decision: Literal['correct', 'wrong'] | None = None


class SheetScore(BaseModel):
    sheet_id: str
    student: str
    identification: Literal['matched', 'review']
    identification_reason: str
    score: int = Field(description='확정 점수')
    max_score: int
    review_count: int = Field(description='교사 확인 문항 수')
    pending_points: int = Field(description='교사 확인 결과에 따라 더해질 수 있는 최대 점수')
    status: Literal['confirmed', 'needs_review']


class MissingStudent(BaseModel):
    """자동 매칭된 답안지가 없는 학생"""
    class_no: int
    number: int
    name: str
    candidate_sheets: list[str] = Field(description='이름이 같아 이 학생 것일 수 있는 교사 확인 답안지')


class Summary(BaseModel):
    sheets: int
    answers: int
    correct: int
    wrong: int
    review: int
    auto_rate: float = Field(description='교사가 보지 않아도 되는 답안 비율')
    identification_review: int = Field(description='학생 식별을 교사가 확인해야 하는 답안지 수')
    confirmed_sheets: int
    teacher_decided: int = Field(0, description='선생님이 정답이나 오답으로 정한 확인 문항 수')


class GradeResponse(BaseModel):
    summary: Summary
    answers: list[AnswerResult]
    review_items: list[ReviewItem] = Field(description='교사 확인 목록 (노트북의 교사확인목록.csv)')
    scores: list[SheetScore] = Field(description='성적표 (노트북의 성적표.csv)')
    missing_students: list[MissingStudent]
    warnings: list[str]
    policy: PolicyIn


class HealthResponse(BaseModel):
    status: Literal['ok']
    model: dict
