"""채점된 시험지: 빨간펜으로 문항 번호에 표시 (노트북 4-8)

정답 = 번호에 동그라미, 오답 = 번호에 빗금, 교사 확인 = 번호 왼쪽에 "(재확인)", 점수 칸에 확정 점수.
"""
import random

from PIL import Image, ImageDraw

from .images import PEN_FONT, font
from .layout import PX, PY, SCORE_BOX

PEN = (214, 30, 36, 235)
NUMBER_MARKS = {1: (0.086, 0.315, 0.100, 0.332), 2: (0.086, 0.435, 0.100, 0.451), 3: (0.086, 0.554, 0.100, 0.570),
                4: (0.086, 0.673, 0.100, 0.690), 5: (0.086, 0.792, 0.100, 0.809)}   # 시험지 PDF에서 잰 문항 번호 위치


def mark_sheet(sid, flat_path, finals, score, review_count, identified, out_path):
    """finals: {문항: 'correct' | 'wrong' | 'review'} → 빨간펜 채점 이미지 저장 후 반환"""
    page = Image.open(flat_path).convert('RGBA')
    layer = Image.new('RGBA', page.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rng = random.Random(sid)   # 같은 시험지는 항상 같은 손맛
    pen = lambda size: font(PEN_FONT, size)
    for q, final in finals.items():
        if q not in NUMBER_MARKS:
            continue
        x1, y1, x2, y2 = (PX(v) if k % 2 == 0 else PY(v) for k, v in enumerate(NUMBER_MARKS[q]))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if final == 'correct':        # 번호에 동그라미 (두 번 그려 손으로 그린 느낌)
            for _ in range(2):
                rr = 42 + rng.uniform(-3, 3)
                o = [rng.uniform(-4, 4) for _ in range(2)]
                d.ellipse((cx - rr + o[0], cy - rr * 0.92 + o[1], cx + rr + o[0], cy + rr * 0.92 + o[1]), outline=PEN, width=6)
        elif final == 'wrong':        # 번호에 빗금
            d.line((cx - 34, cy + 40, cx + 30, cy - 42), fill=PEN, width=8)
        else:                         # 번호 왼쪽에 (재확인)
            size = 40
            while size > 14 and d.textlength('(재확인)', font=pen(size)) > x1 - 10:
                size -= 2
            d.text((x1 - 6 - d.textlength('(재확인)', font=pen(size)), cy - size * 0.6), '(재확인)', font=pen(size), fill=PEN)
    sx1, sy1, sx2, sy2 = PX(SCORE_BOX[0]), PY(SCORE_BOX[1]), PX(SCORE_BOX[2]), PY(SCORE_BOX[3])
    txt = f'{score}'
    d.text(((sx1 + sx2 - d.textlength(txt, font=pen(72))) / 2, sy1 - 10), txt, font=pen(72), fill=PEN)
    if review_count:
        sub = f'+확인 {review_count}문항'
        d.text(((sx1 + sx2 - d.textlength(sub, font=pen(28))) / 2, sy2 - 30), sub, font=pen(28), fill=PEN)
    if not identified:                # 제목 왼쪽 빈 곳
        d.text((PX(0.095), PY(0.110)), '학생 확인 요함', font=pen(36), fill=PEN)
    out = Image.alpha_composite(page, layer).convert('RGB')
    out.save(out_path)
    return out
