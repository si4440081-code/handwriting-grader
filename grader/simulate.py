"""학습 데이터 만들기 (노트북 2단계)

실제 학생 답안지를 쓸 수 없으므로, 학생이 쓴 답과 손글씨를 읽을 때 생기는 실수를 흉내 내서 만든다.
- 학생 답 유형: 정답 45%, 등록오답 20%, 혼동쌍 오타 12%, 일반 오타 8%, 순서 바뀜 8%, 빈칸 7% (KINDS)
- 읽기 실수: 글씨가 나쁠수록 혼동쌍으로 잘못 읽음 + 정답과 한 글자 차이면 정답으로 고쳐 읽는 실수
- 읽는 방식은 4단계와 같은 적응형: 2번 읽어 같으면 끝, 다르면 3번째까지

노트북은 전역 난수(random.seed, np.random.seed)를 쓰지만, 서버는 여러 과목을 학습하므로 시드마다 난수 객체를 따로 만든다.
같은 시드면 노트북과 같은 순서로 난수를 뽑으므로 같은 데이터가 나온다.
"""
import random

import numpy as np

from .answer_key import edit_distance, normalize
from .features import FEATURES, make_features
from .hangul import CHO, CONFUSABLE, JONG, JUNG, SIMPLE_CONFUSABLE, compose, decompose
from .judge import near_miss

MIN_READS, MAX_READS = 2, 3


def confusion_variants(ch):
    """글자 하나를 혼동쌍으로 잘못 읽었을 때 나올 수 있는 글자들"""
    if ch in SIMPLE_CONFUSABLE:
        return list(SIMPLE_CONFUSABLE[ch])
    d = decompose(ch)
    if d is None:
        return []
    cho, jung, jong = d
    out = set()
    for a, b in CONFUSABLE:
        for x, y in ((a, b), (b, a)):
            if cho == x and y in CHO: out.add(compose(y, jung, jong))
            if jung == x and y in JUNG: out.add(compose(cho, y, jong))
            if jong == x and y in JONG: out.add(compose(cho, jung, y))
    if jong:
        out.add(compose(cho, jung, ''))
    else:
        out.update(compose(cho, jung, j) for j in ('ㄴ', 'ㅇ'))
    out.discard(ch)
    return sorted(out)


# 학생 답 유형별로 답 하나를 만드는 함수 → (답, 정답여부). 그 문항으로 만들 수 없으면 IndexError
def make_correct(q, rnd):
    return rnd.choice([q.answer, *q.allowed]), 1
def make_wrong(q, rnd):              # 그럴듯한 오답: 정답표의 등록오답 열에서만 가져옴
    return rnd.choice(q.registered_wrong), 0
def make_confusion_typo(q, rnd):     # 훈민정읍, DNA → DNR 같은 혼동쌍 오타
    base = rnd.choice(q.accepted)
    i = rnd.choice([i for i, c in enumerate(base) if confusion_variants(c)])
    return base[:i] + rnd.choice(confusion_variants(base[i])) + base[i + 1:], 0
def make_typo(q, rnd):               # 이순산, photosynthasis 같은 일반 오타
    base = rnd.choice(q.accepted)
    i = rnd.choice([i for i, c in enumerate(base) if decompose(c) or c.isascii() and c.isalpha()])
    c = compose(rnd.choice(CHO), rnd.choice(JUNG)) if decompose(base[i]) \
        else rnd.choice([x for x in 'abcdefghijklmnopqrstuvwxyz' if x != base[i]])
    return base[:i] + c + base[i + 1:], 0
def make_shuffle(q, rnd):            # 이신순
    base = rnd.choice([a for a in q.accepted if len(set(a)) > 1])
    t = base
    while t == base:
        t = ''.join(rnd.sample(base, len(base)))
    return t, 0
def make_blank(q, rnd):
    return '', 0

KINDS = {'정답': (make_correct, 0.45), '그럴듯한오답': (make_wrong, 0.20), '혼동쌍오타': (make_confusion_typo, 0.12),
         '일반오타': (make_typo, 0.08), '순서바뀜': (make_shuffle, 0.08), '빈칸': (make_blank, 0.07)}


def student_answer(q, rnd):
    while True:
        kind = rnd.choices(list(KINDS), weights=[w for _, w in KINDS.values()])[0]
        try:
            text, label = KINDS[kind][0](q, rnd)
            break
        except IndexError:   # 이 문항으로는 만들 수 없는 유형 (영어 정답의 한글 혼동쌍, 등록오답 없음 등) → 다시 고름
            continue
    if normalize(text) in q.accepted:   # 우연히 허용 답안이 되면 정답으로
        label = 1
    return text, label, kind


def simulate_reads(text, q, rnd, nprnd):
    """같은 답안을 읽는 것을 흉내 낸다 (적응형: 2번 읽어 같으면 끝, 다르면 3번째까지).
    글씨가 나쁠수록 혼동쌍으로 잘못 읽고, 정답과 한 글자 차이면 정답으로 '고쳐 읽는' 실수도 한다."""
    text = normalize(text)
    bad = nprnd.beta(2, 5)                             # 글씨 상태: 0 깔끔 ~ 1 악필
    messy = rnd.random() < 0.06                        # 지운 자국 등
    err = 0.01 + 0.25 * bad + (0.2 if messy else 0)    # 글자 하나를 잘못 읽을 확률
    near = [a for a in q.accepted if len(a) == len(text) and edit_distance(text, a) == 1]
    reads = []
    for k in range(MAX_READS):
        if near and rnd.random() < 0.1 + 0.3 * bad:
            reads.append(rnd.choice(near))
        else:
            reads.append(''.join(rnd.choice(confusion_variants(c)) if rnd.random() < err and confusion_variants(c) else c
                                 for c in text))
        if k + 1 >= MIN_READS and len(set(reads)) == 1:
            break
    return reads, messy


def make_dataset(answer_key, n_samples, seed):
    """시뮬레이션 데이터 n개 → X (n × 8), y (n,), 행별 정보 [{문항, 학생답, 유형, 읽기, 지저분함, 근처}]"""
    rnd, nprnd = random.Random(seed), np.random.RandomState(seed)
    rows, info = [], []
    for _ in range(n_samples):
        q = answer_key[rnd.choice(list(answer_key))]
        text, label, kind = student_answer(q, rnd)
        reads, messy = simulate_reads(text, q, rnd, nprnd)
        f = make_features(reads, q, messy)
        rows.append([label] + [f[k] for k in FEATURES])
        info.append({'문항': q.number, '학생답': normalize(text), '유형': kind, '읽기': reads, '지저분함': messy,
                     '근처': near_miss(reads, q)})
    data = np.array(rows, dtype=float)
    return data[:, 1:], data[:, 0].astype(int), info
