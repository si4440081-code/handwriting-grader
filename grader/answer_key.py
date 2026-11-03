"""정답표(허용 답안 사전)와 답 비교에 쓰는 문자열 도구 (노트북 1-4)"""
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from functools import cached_property

CASE_SENSITIVE = False   # 영어 대소문자 구분 (False면 DNA = dna)


def normalize(s):
    """띄어쓰기 제거, 마침표류를 가운뎃점으로 통일, (설정에 따라) 영어 소문자로
    ("이순신 장군" = "이순신장군", "3.1운동" = "3·1 운동", "DNA" = "dna")"""
    s = unicodedata.normalize('NFC', str(s or ''))
    s = re.sub(r'\s+', '', s)
    for ch in '.．ㆍ•∙⋅・':
        s = s.replace(ch, '·')
    return s if CASE_SENSITIVE else s.lower()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class Question:
    """정답표 한 줄: 교사가 문항별로 적은 정답, 함께 인정할 답, 확실한 오답, 배점"""
    number: int
    answer: str
    points: int = 20
    allowed: tuple[str, ...] = ()
    registered_wrong: tuple[str, ...] = ()
    memo: str = ''

    @cached_property
    def accepted(self):
        """인정하는 답 전부 (띄어쓰기와 기호를 정리한 뒤, 중복 없이 순서 유지)"""
        return list(dict.fromkeys(a for a in map(normalize, (self.answer, *self.allowed)) if a))

    @cached_property
    def registered_wrong_normalized(self):
        return {normalize(w) for w in self.registered_wrong}


def fingerprint(answer_key):
    """정답표 지문 (노트북 3-7과 같은 계산). 노트북에서 내보낸 grader.json이 어느 정답표로 학습됐는지 맞춰 본다"""
    items = sorted((q.number, q.answer, tuple(q.allowed), tuple(q.registered_wrong)) for q in answer_key.values())
    return hashlib.sha1(repr(items).encode()).hexdigest()[:12]
