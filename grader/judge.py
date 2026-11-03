"""신경망 확률 + 교사 정책 규칙 → 판정 (노트북 3-3, 프로젝트 정리 문서 4절과 5절)"""
from dataclasses import dataclass

from .answer_key import edit_distance
from .features import top_read

CORRECT, WRONG, REVIEW = 'correct', 'wrong', 'review'   # 정답, 오답, 교사 확인
MIXED_READS_RULES = ('strict', 'lenient', 'off')


@dataclass(frozen=True)
class Policy:
    """교사가 정하는 판정 기준. 기본값은 노트북 1-2 설정과 같다."""
    t_high: float = 0.8                # P(정답) ≥ t_high → 정답
    t_low: float = 0.2                 # P(정답) ≤ t_low → 오답, 그 사이 → 교사 확인
    mixed_reads: str = 'strict'        # 읽기가 엇갈릴 때: 'strict' 하나라도 엇갈리면 교사 확인 / 'lenient' 80% 미만이면 / 'off' 확률에 맡김
    near_miss_to_review: bool = True   # 한 글자 차이이거나 정답 뒤에 말이 붙은 답은 교사 확인

    def __post_init__(self):
        if not 0 <= self.t_low < self.t_high <= 1:
            raise ValueError(f't_low({self.t_low})는 t_high({self.t_high})보다 작고, 둘 다 0~1 사이여야 합니다')
        if self.mixed_reads not in MIXED_READS_RULES:
            raise ValueError(f'mixed_reads는 {MIXED_READS_RULES} 중 하나여야 합니다')


def near_miss(reads, question):
    """한 글자 차이이거나 정답 뒤에 말이 붙은 답이면 True (교사가 등록한 오답은 제외: "평양"과 "한양").
    쉼표로 답을 두 개 이상 쓴 경우는 'two'"""
    top, _ = top_read(reads)
    if ',' in top or '，' in top:
        return 'two'
    if not top or top in question.accepted or top in question.registered_wrong_normalized:
        return False
    return any(edit_distance(top, a) == 1 or top.startswith(a) for a in question.accepted)


def rule_check(f, policy, outside=False, near=False):
    """확률과 상관없이 교사 확인으로 보내는 규칙 (정리 문서 4절 판정 순서). 걸리면 사유, 아니면 None"""
    r = f['정답읽힘비율']
    if outside:
        return '칸 밖에 쓴 글씨가 있음'
    if near == 'two':
        return '답을 두 개 이상 씀 (어느 답을 채점할지 교사가 결정)'
    if f['지저분함']:
        return '칸이 지저분함 (지운 자국, 답 두 개 등)'
    if f['순서만다름']:
        return '글자는 같고 순서만 다름 (어순 정책은 교사가 결정)'
    if f['혼동쌍만다름']:
        return '혼동쌍만 다름 (학생 오타인지 글씨 문제인지 구분 불가)'
    if (policy.mixed_reads == 'strict' and 0 < r < 1) or (policy.mixed_reads == 'lenient' and 0 < r < 0.8):
        return '읽기 결과가 엇갈림: 일부만 정답으로 읽힘 (어느 읽기를 믿느냐에 따라 판정이 바뀜)'
    if policy.near_miss_to_review and near:
        return '한 글자 차이이거나 정답 뒤에 말이 붙음 (오타 정책은 교사가 결정)'
    return None


def decide(p, f, policy, outside=False, near=False):
    """신경망 확률(p) + 교사 정책 규칙 → (판정, 사유)"""
    reason = rule_check(f, policy, outside, near)
    if reason:
        return REVIEW, reason
    if p >= policy.t_high:
        return CORRECT, f'P(정답) {p:.0%} ≥ {policy.t_high:.0%}'
    if p <= policy.t_low:
        return WRONG, f'P(정답) {p:.0%} ≤ {policy.t_low:.0%}'
    return REVIEW, f'P(정답) {p:.0%}: 애매한 구간'
