"""이미지 파일 읽기, 쓰기와 글꼴 (한글 경로에서도 동작하도록 OpenCV 대신 numpy로 파일을 다룬다)"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

FONT_DIR = Path(__file__).resolve().parent.parent / 'fonts'
PEN_FONT = FONT_DIR / 'NanumPenScript-Regular.ttf'    # 빨간펜 글씨 (나눔펜, SIL 오픈 폰트 라이선스)
GOTHIC_FONT = FONT_DIR / 'NanumGothic-Regular.ttf'    # AI에게 보내는 칸 이름표


@lru_cache(maxsize=None)
def font(path, size):
    return ImageFont.truetype(str(path), size)


def imread(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'이미지를 읽을 수 없습니다: {Path(path).name}')
    return img


def imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or '.png', img)
    if not ok:
        raise ValueError(f'이미지를 저장할 수 없습니다: {Path(path).name}')
    buf.tofile(str(path))
