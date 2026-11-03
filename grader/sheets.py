"""답안지와 답안 칸 (노트북 4-4)"""
from dataclasses import dataclass, field


@dataclass
class Cell:
    """답안 칸 하나: 여러 번 읽은 결과와 칸 상태"""
    reads: list[str]
    messy: bool = False     # 지운 자국, 답 두 개 등
    outside: bool = False   # 칸 밖에 쓴 글씨
    note: str = ''          # AI가 읽지 못한 칸의 사유. 있으면 확률과 상관없이 교사 확인
    image: str = ''         # 잘라 낸 칸 이미지 경로 (사진으로 채점할 때)


@dataclass
class Sheet:
    """답안지 한 장. 반과 번호는 손글씨에서 읽은 값이라 비어 있거나 숫자가 아닐 수 있어 문자열로 받는다."""
    sheet_id: str
    class_no: str
    number: str
    name: str
    answers: dict[int, Cell] = field(default_factory=dict)

    @property
    def student(self):
        return f'{self.class_no}반 {self.number}번 {self.name}'


def parse_cell(cell):
    """답안 칸 표시 해석: 앞에 * 지저분함, 앞에 ! 칸 밖, | 로 구분 여러 번 읽은 결과 → Cell"""
    s, messy, outside = str(cell), False, False
    while s[:1] in ('*', '!'):
        messy, outside = messy or s[0] == '*', outside or s[0] == '!'
        s = s[1:]
    return Cell([t.strip() for t in s.split('|')], messy, outside)
