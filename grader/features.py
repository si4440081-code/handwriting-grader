"""읽기 결과 → 신경망 입력 특징 8개 (노트북 1-5)

| 특징 | 뜻 |
|---|---|
| 정답읽힘비율 | 읽기 결과 중 허용 답안과 같은 비율 |
| 읽기일치율 | 읽기 결과끼리 같은 비율 |
| 글자유사도 | 가장 가까운 허용 답안과 얼마나 비슷한지 (1이면 같음) |
| 길이차이 | 허용 답안과 글자 수 차이 |
| 혼동쌍만다름 | ㅁ/ㅂ, ㅗ/ㅜ 같은 헷갈리는 글자만 다른지 (예: 훈민정읍). 등록한 오답은 제외 |
| 순서만다름 | 글자는 같고 순서만 다른지 (예: 이신순). 등록한 오답은 제외 |
| 빈칸 | 아무것도 안 썼는지 |
| 지저분함 | 지운 자국, 답 두 개 등 |
"""
from collections import Counter

from .answer_key import edit_distance, normalize
from .hangul import is_confusable

FEATURES = ['정답읽힘비율', '읽기일치율', '글자유사도', '길이차이', '혼동쌍만다름', '순서만다름', '빈칸', '지저분함']


def top_read(texts):
    """가장 많이 나온 읽기 결과(정리한 뒤)와 그 비율"""
    top, n = Counter(normalize(t) for t in texts).most_common(1)[0]
    return top, n / len(texts)


def make_features(texts, question, messy=False):
    """같은 답안을 여러 번 읽은 결과(texts) → 신경망에 넣을 숫자 8개"""
    texts = [normalize(t) for t in texts]
    accepted = question.accepted
    ok = set(accepted)
    top, agree = top_read(texts)                               # 가장 많이 나온 읽기 결과
    best = min(accepted, key=lambda a: edit_distance(top, a))  # 가장 가까운 허용 답안
    d = edit_distance(top, best)
    mism = [(x, y) for x, y in zip(top, best) if x != y] if len(top) == len(best) else []
    registered = top in question.registered_wrong_normalized   # 교사가 등록한 오답은 혼동쌍, 순서 규칙에 걸리지 않게 해서 그대로 오답
    return {
        '정답읽힘비율': sum(t in ok for t in texts) / len(texts),   # 허용 답안으로 읽힌 비율
        '읽기일치율': agree,                                        # 읽을 때마다 같은 결과가 나온 비율
        '글자유사도': 1 - d / max(len(top), len(best), 1),          # 1이면 허용 답안과 똑같음
        '길이차이': min(abs(len(top) - len(best)), 3) / 3,
        '혼동쌍만다름': float(not registered and bool(mism) and all(is_confusable(x, y) for x, y in mism)),  # 예: 훈민정읍
        '순서만다름': float(not registered and top != best and sorted(top) == sorted(best)),               # 예: 이신순
        '빈칸': float(top == ''),
        '지저분함': float(bool(messy)),                             # 지운 자국, 답 두 개 등
    }
