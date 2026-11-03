"""신경망 학습 → models/grader.json (노트북 2단계 끝, 3-1, 3-7)

    python -m training.train                       # 예시 역사 정답표로 학습
    python -m training.train --answer-key data/subjects/영어/영어_시험자료.xlsx --out models/grader_en.json

노트북과 같은 Keras 모델을 TensorFlow로 학습하고, 텍스트 채점 주소(/v1/grade)가 쓰는 models/grader.json을 만든다.
시연 서버는 과목마다 grader/trainer.py(numpy)로 직접 학습하므로 이 스크립트가 필요 없다.
여기서 나오는 정확도와 자동 처리 오류는 시뮬레이션 데이터 기준이다. 실제 손글씨에서의 성능이 아니다.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from grader.answer_key import fingerprint
from grader.features import FEATURES
from grader.judge import CORRECT, REVIEW, Policy, decide
from grader.model import DEFAULT_MODEL_PATH, GraderModel
from grader.simulate import make_dataset
from grader.tables import ANSWER_KEY, classify, parse_answer_key, read_table_file

ROOT = Path(__file__).resolve().parent.parent


def ece(p, y, bins=10):
    """보정 오차: 확률 구간마다 |말한 확률 평균 - 실제 정답률|을 구간 크기로 가중 평균"""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def auto_stats(p, X, y, info, policy):
    """시험용 데이터를 판정했을 때 자동 처리 비율과 자동 처리 중 틀린 판정 수"""
    decisions = [decide(round(float(pi), 3), dict(zip(FEATURES, x)), policy, near=row['근처'])[0]
                 for pi, x, row in zip(p, X, info)]
    auto = [(d, yi) for d, yi in zip(decisions, y) if d != REVIEW]
    errors = sum((d == CORRECT) != (yi == 1) for d, yi in auto)
    return {'auto_rate': round(len(auto) / len(y), 3), 'auto_errors': int(errors), 'auto_total': len(auto)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--answer-key', default=ROOT / 'data' / 'subjects' / '역사' / '역사_시험자료.xlsx', type=Path)
    ap.add_argument('--samples', default=3000, type=int)
    ap.add_argument('--seed', default=42, type=int)
    ap.add_argument('--out', default=DEFAULT_MODEL_PATH, type=Path)
    args = ap.parse_args()

    found, _ = classify(read_table_file(args.answer_key))
    answer_key = parse_answer_key(found[ANSWER_KEY])
    tf.keras.utils.set_random_seed(args.seed)
    X, y, info = make_dataset(answer_key, args.samples, args.seed)
    X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=args.seed, stratify=y)
    print(f'학습 데이터 {len(y)}개 (학습용 {len(y_train)} / 시험용 {len(y_test)})')

    model = Sequential([Input(shape=(len(FEATURES),)), Dense(10, activation='relu'), Dense(1, activation='sigmoid')])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, epochs=60, batch_size=64, validation_split=0.2, verbose=0,
                     callbacks=[EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)]).history
    best = int(np.argmin(hist['val_loss']))
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    grader = GraderModel([{'activation': layer.get_config()['activation'],
                           'kernel': layer.get_weights()[0], 'bias': layer.get_weights()[1]}
                          for layer in model.layers])
    # API 서버의 numpy 계산이 Keras와 같은 값을 내는지 확인한 뒤에만 저장한다
    p_numpy = grader.predict(X_test)
    diff = float(np.abs(model.predict(X_test, verbose=0).ravel() - p_numpy).max())
    if diff > 1e-5:
        raise SystemExit(f'numpy 추론이 Keras와 다릅니다 (최대 차이 {diff})')

    test_info = [info[i] for i in i_test]
    grader.info = {
        'source': 'training_script', 'answer_key': fingerprint(answer_key),   # 노트북 3-7과 같은 정답표 지문
        'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'tensorflow': tf.__version__,
        'seed': args.seed, 'samples': args.samples, 'epochs': len(hist['loss']),
        'val_accuracy': round(float(hist['val_accuracy'][best]), 4),
        'simulation_test': {   # 시뮬레이션 시험용 데이터 기준 (실제 손글씨 성능이 아님)
            'size': len(y_test), 'accuracy': round(acc, 4), 'loss': round(loss, 4), 'ece': round(ece(p_numpy, y_test), 4),
            'default_policy': auto_stats(p_numpy, X_test, y_test, test_info, Policy()),
            'probability_only': auto_stats(p_numpy, X_test, y_test, test_info,
                                           Policy(mixed_reads='off', near_miss_to_review=False)),
        },
        'numpy_vs_keras_max_diff': diff,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grader.save(args.out)
    print(f'저장: {args.out}')
    print(grader.info)


if __name__ == '__main__':
    main()
