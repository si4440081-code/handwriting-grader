"""신경망 학습을 numpy로 (노트북 3-1의 Keras 학습과 같은 방식)

시연 서버는 새 정답표를 받으면 그 자리에서 다시 학습해야 하는데, TensorFlow를 서버에 넣으면 무겁고 느리게 뜬다.
그래서 노트북 3-1과 같은 구조와 학습 방식을 numpy로 옮겼다.
- 구조: Dense(10, relu) → Dense(1, sigmoid), 가중치 초기값은 Keras 기본값(Glorot uniform, 편향 0)
- 학습: binary cross-entropy, Adam(학습률 0.001), batch_size 64, 최대 60 에포크
- validation_split 0.2: Keras처럼 데이터 끝 20%를 검증용으로 씀
- 조기 종료: 검증 loss가 10 에포크 동안 0.001 넘게 줄지 않으면 멈추고 가장 좋았던 가중치로 되돌림
"""
import numpy as np

from .features import FEATURES
from .model import GraderModel

EPOCHS, BATCH_SIZE, HIDDEN = 60, 64, 10
PATIENCE, MIN_DELTA, VALIDATION_SPLIT = 10, 0.001, 0.2
LR, BETA1, BETA2, EPS = 0.001, 0.9, 0.999, 1e-7


def _glorot(rng, fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out))


def _forward(params, X):
    W1, b1, W2, b2 = params
    h = np.maximum(X @ W1 + b1, 0)
    z = (h @ W2 + b2).ravel()
    return h, z, 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def _loss_acc(params, X, y):
    p = np.clip(_forward(params, X)[2], 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), float(np.mean((p > 0.5) == y))


def train(X, y, seed=42, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """특징 X (n × 8), 정답 y (n,) → GraderModel (info에 에포크 수, 검증 정확도)"""
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    split = int(round(len(X) * (1 - VALIDATION_SPLIT)))
    Xt, yt, Xv, yv = X[:split], y[:split], X[split:], y[split:]
    params = [_glorot(rng, len(FEATURES), HIDDEN), np.zeros(HIDDEN), _glorot(rng, HIDDEN, 1), np.zeros(1)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    best = (np.inf, None, 0, 0.0)   # (검증 loss, 가중치, 에포크, 검증 정확도)
    wait, epoch = 0, 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(Xt))
        for start in range(0, len(Xt), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = Xt[idx], yt[idx]
            h, _, p = _forward(params, xb)
            dz = ((p - yb) / len(idx))[:, None]          # sigmoid + binary cross-entropy의 기울기
            dh = (dz @ params[2].T) * (h > 0)
            grads = [xb.T @ dh, dh.sum(0), h.T @ dz, dz.sum(0)]
            t += 1
            for i, g in enumerate(grads):
                m[i] = BETA1 * m[i] + (1 - BETA1) * g
                v[i] = BETA2 * v[i] + (1 - BETA2) * g * g
                lr_t = LR * np.sqrt(1 - BETA2 ** t) / (1 - BETA1 ** t)
                params[i] = params[i] - lr_t * m[i] / (np.sqrt(v[i]) + EPS)
        val_loss, val_acc = _loss_acc(params, Xv, yv)
        if val_loss < best[0] - MIN_DELTA:
            best, wait = (val_loss, [p.copy() for p in params], epoch, val_acc), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    W1, b1, W2, b2 = best[1]
    return GraderModel([{'activation': 'relu', 'kernel': W1, 'bias': b1},
                        {'activation': 'sigmoid', 'kernel': W2, 'bias': b2}],
                       {'epochs': epoch, 'best_epoch': best[2], 'val_loss': round(best[0], 4),
                        'val_accuracy': round(best[3], 4)})
