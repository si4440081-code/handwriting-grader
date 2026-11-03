"""학습한 신경망으로 P(정답)을 계산한다 (노트북 3단계 모델의 추론 부분)

학습은 training/train.py가 TensorFlow(Keras Sequential)로 하고, 가중치만 JSON으로 저장한다.
API 서버는 같은 계산(Dense 층마다 X·W + b → 활성화 함수)을 numpy로 해서 TensorFlow 없이 돌아간다.
"""
import json
from pathlib import Path

import numpy as np

from .features import FEATURES

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'grader.json'

ACTIVATIONS = {
    'relu': lambda x: np.maximum(x, 0),
    'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
    'linear': lambda x: x,
}


class GraderModel:
    def __init__(self, layers, info=None):
        """layers: [{'activation': 'relu', 'kernel': [[...]], 'bias': [...]}, ...] (입력 쪽 층부터)"""
        self.layers = [(np.asarray(l['kernel'], dtype=np.float64), np.asarray(l['bias'], dtype=np.float64),
                        l['activation']) for l in layers]
        self.info = info or {}
        if self.layers[0][0].shape[0] != len(FEATURES) or self.layers[-1][0].shape[1] != 1:
            raise ValueError(f'모델 입력은 특징 {len(FEATURES)}개, 출력은 1개여야 합니다')
        for (w, b, _), (w_next, _, _) in zip(self.layers, self.layers[1:] + [(np.zeros((1, 1)), None, None)]):
            if w.ndim != 2 or b.shape != (w.shape[1],) or w.shape[1] != w_next.shape[0]:
                raise ValueError('모델 층의 가중치 크기가 서로 맞지 않습니다')
        unknown = {a for _, _, a in self.layers} - set(ACTIVATIONS)
        if unknown:
            raise ValueError(f'지원하지 않는 활성화 함수: {unknown}')

    @classmethod
    def from_dict(cls, data):
        """grader.json 내용 → 모델. 노트북 3-7에서 내보낸 파일도 같은 형식이다"""
        if not isinstance(data, dict) or not isinstance(data.get('layers'), list) or not data['layers']:
            raise ValueError('grader.json 형식이 아닙니다 (features, layers가 있어야 합니다)')
        if data.get('features') != FEATURES:   # 특징 순서가 다르면 입력이 엉뚱한 자리에 들어간다
            raise ValueError(f'모델의 특징 목록이 코드와 다릅니다: {data.get("features")}')
        try:
            return cls(data['layers'], data.get('info'))
        except (KeyError, TypeError, IndexError) as e:
            raise ValueError(f'grader.json의 가중치를 읽을 수 없습니다 ({type(e).__name__})') from None

    @classmethod
    def load(cls, path=DEFAULT_MODEL_PATH):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    def to_dict(self):
        return {
            'features': FEATURES,
            'layers': [{'activation': a, 'kernel': w.tolist(), 'bias': b.tolist()} for w, b, a in self.layers],
            'info': self.info,
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding='utf-8')

    def predict(self, X):
        """특징 행렬 (n × 8) → P(정답) (n,)"""
        h = np.asarray(X, dtype=np.float64).reshape(-1, len(FEATURES))
        for w, b, activation in self.layers:
            h = ACTIVATIONS[activation](h @ w + b)
        return h.ravel()
