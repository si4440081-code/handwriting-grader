"""정렬과 칸 자르기 (노트북 4-2)

네 모서리 검은 사각형으로 사진을 반듯하게 펴고, 반, 번호, 이름 칸과 정답 칸을 자른다.
칸 밖에 쓴 글씨가 있으면 표시하고 주변까지 넓게 잘라 둔다. 좌표는 시험지 PDF에서 잰 비율이다.
"""
import json
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .images import imread, imwrite

CORNERS = np.float32([[0.05, 0.0354], [0.95, 0.0354], [0.05, 0.9646], [0.95, 0.9646]])  # 네 모서리 검은 사각형 중심
ID_BOX = (0.430, 0.160, 0.810, 0.210)                                                   # '( )반 ( )번 이름( )'
SCORE_BOX = (0.762, 0.101, 0.886, 0.155)                                                # 점수 칸
ANSWER_BOXES = {1: (0.190, 0.365, 0.667, 0.419), 2: (0.190, 0.484, 0.667, 0.538), 3: (0.190, 0.603, 0.667, 0.657),
                4: (0.190, 0.723, 0.667, 0.776), 5: (0.190, 0.842, 0.667, 0.896)}
AROUND_BOXES = {1: (0.175, 0.357, 0.950, 0.431), 2: (0.175, 0.476, 0.950, 0.550), 3: (0.175, 0.596, 0.950, 0.669),
                4: (0.175, 0.715, 0.950, 0.788), 5: (0.175, 0.834, 0.950, 0.938)}   # 칸 주변의 원래 빈 영역
PAGE_W, PAGE_H, INSET = 1240, 1754, 0.006
OUTSIDE_MIN_INK = 50   # 칸 밖 잉크 픽셀이 이보다 많으면 '칸 밖 글씨'
PX = lambda v: int(v * PAGE_W)
PY = lambda v: int(v * PAGE_H)


@dataclass
class SheetCrops:
    """한 장을 자른 결과 (경로는 모두 같은 폴더 안)"""
    flat: str                 # 반듯하게 편 시험지 전체
    id: str                   # 반, 번호, 이름 칸
    cells: dict[int, str]     # 문항 → 정답 칸 (칸 밖 글씨가 있으면 주변까지 넓게 자른 것)
    outside: dict[int, bool]  # 문항 → 칸 밖 글씨가 있는지
    aligned: bool             # 모서리 표시를 찾아 반듯하게 폈는지


def find_corners(gray):
    """네 모서리 검은 사각형 중심 [좌상, 우상, 좌하, 우하]. 그림자에 강하게 전역, 지역 이진화를 둘 다 쓰고,
    3개만 찾으면(그림자에 묻히거나 사진 끝에 잘리면) 나머지 하나를 추정"""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, (max(w, h) // 15) | 1, 25)
    cands = []
    for bw in (otsu, adapt):
        for c in cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, cw, ch = cv2.boundingRect(c)
            if (w * 0.008 < cw < w * 0.06 and w * 0.008 < ch < w * 0.06 and 0.5 < cw / ch < 2.0
                    and cv2.contourArea(c) >= 0.75 * cw * ch and blur[y:y + ch, x:x + cw].mean() <= 110):
                cands.append((x + cw / 2, y + ch / 2))
    found = {}
    for k, (tx, ty) in enumerate(((0, 0), (w, 0), (0, h), (w, h))):
        side = [p for p in cands if (p[0] < w / 2) == (tx == 0) and (p[1] < h / 2) == (ty == 0)]
        if side:
            found[k] = min(side, key=lambda p: np.hypot(p[0] - tx, p[1] - ty))
    if len(found) == 3:
        miss = ({0, 1, 2, 3} - set(found)).pop()
        a, b, c = {0: (1, 2, 3), 1: (0, 3, 2), 2: (0, 3, 1), 3: (1, 2, 0)}[miss]
        found[miss] = tuple(np.add(found[a], found[b]) - np.array(found[c]))
    return np.float32([found[k] for k in range(4)]) if len(found) == 4 else None


def ink_outside(flat, q):
    """정답 칸 바깥 주변에 글씨(잉크)가 있는지. 그림자에 강하게, 칸 테두리 같은 가는 직선은 제외 → 잉크 픽셀 수"""
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    ax1, ay1, ax2, ay2 = AROUND_BOXES[q]
    bx1, by1, bx2, by2 = ANSWER_BOXES[q]
    ink = cv2.adaptiveThreshold(gray[PY(ay1):PY(ay2), PX(ax1):PX(ax2)], 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 35, 25)
    m = 0.013   # 사진 정렬 오차를 감안한 칸 테두리 여유
    ink[max(PY(by1 - m) - PY(ay1), 0):PY(by2 + m) - PY(ay1), max(PX(bx1 - m) - PX(ax1), 0):PX(bx2 + m) - PX(ax1)] = 0
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(ink)
    return int(sum(st[i, 4] for i in range(1, n)
                   if st[i, 4] >= 25 and not (min(st[i, 2:4]) <= 6 and max(st[i, 2:4]) >= 8 * min(st[i, 2:4]))))


def crop_sheet(path, out_dir):
    """사진을 반듯하게 펴서 저장하고, 반, 번호, 이름 칸과 정답 칸을 자르고, 칸 밖 글씨를 검사 → SheetCrops.
    out_dir에 이미 자른 결과가 있으면 다시 자르지 않는다 (예시 답안지)"""
    record = os.path.join(out_dir, 'crops.json')
    if os.path.exists(record):
        data = json.load(open(record, encoding='utf-8'))
        return SheetCrops(data['flat'], data['id'], {int(k): v for k, v in data['cells'].items()},
                          {int(k): v for k, v in data['outside'].items()}, data['aligned'])
    os.makedirs(out_dir, exist_ok=True)
    img = imread(path)
    if max(img.shape[:2]) > 2200:   # 큰 휴대폰 사진은 먼저 줄여서 빠르게 (결과 해상도에는 충분)
        s_ = 2200 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA)
    marks = find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if marks is None:   # 스캔 앱으로 시험지 테두리에 맞춰 잘린 사진이면 크기만 맞춤
        flat = cv2.resize(img, (PAGE_W, PAGE_H))
    else:
        M = cv2.getPerspectiveTransform(marks, CORNERS * np.float32([PAGE_W, PAGE_H]))
        flat = cv2.warpPerspective(img, M, (PAGE_W, PAGE_H), borderValue=(255, 255, 255))
    crops = SheetCrops(os.path.join(out_dir, 'flat.png'), os.path.join(out_dir, 'id.png'), {}, {}, marks is not None)
    imwrite(crops.flat, flat)
    x1, y1, x2, y2 = ID_BOX
    imwrite(crops.id, flat[PY(y1):PY(y2), PX(x1):PX(x2)])
    for q, (x1, y1, x2, y2) in ANSWER_BOXES.items():
        cell = os.path.join(out_dir, f'q{q}.png')
        imwrite(cell, flat[PY(y1 + INSET):PY(y2 - INSET), PX(x1 + INSET):PX(x2 - INSET)])
        crops.outside[q] = ink_outside(flat, q) > OUTSIDE_MIN_INK
        if crops.outside[q]:   # 칸 밖 글씨가 있으면 주변까지 넓게 잘라 둠
            ax1, ay1, ax2, ay2 = AROUND_BOXES[q]
            cell = os.path.join(out_dir, f'q{q}_wide.png')
            imwrite(cell, flat[PY(ay1):PY(ay2), PX(ax1):PX(ax2)])
        crops.cells[q] = cell
    json.dump(asdict(crops), open(record, 'w', encoding='utf-8'), ensure_ascii=False)
    return crops
