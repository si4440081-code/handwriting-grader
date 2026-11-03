"""AI 읽기 (노트북 4-3, 4-4)

Gemini에게 정답을 알려주지 않고 보이는 그대로 읽게 한다. 한 장의 칸들을 이미지 한 장으로 모아 한 번에 읽는다(READ_MODE 'sheet').
- 'safe': 2번 읽어 모든 칸이 같으면 끝, 하나라도 다르면 3번째 (같은 칸을 여러 번 읽은 일치율을 확신도로 씀)
- 'fast': 한 번 읽고, 애매한 칸(허용 답안과 한두 글자 차이, 헷갈리는 글자, 지저분함, 빈칸)이 있을 때만 한 번 더
읽은 결과는 저장해 두고 다시 부르지 않는다. 키가 없어도 미리 읽어 둔 결과(gemini_cache.json)가 있으면 그것을 쓴다.
"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from PIL import Image, ImageDraw

from grader.answer_key import edit_distance, normalize
from grader.hangul import is_confusable

from .images import GOTHIC_FONT, font

SHEET_PROMPT = """이 이미지는 한 학생의 시험 답안지에서 칸들을 잘라 위에서부터 차례로 붙인 것입니다. 각 칸 왼쪽 위에 칸 이름이 적혀 있습니다.
각 칸에 학생이 손으로 쓴 글씨를 보이는 그대로 옮기세요. 정답을 추측하거나 맞춤법을 고치지 마세요.
'(칸 밖 글씨)'라고 적힌 칸은 네모 정답란 밖에도 글씨가 있는 경우이니, X표나 줄을 그어 지운 글씨는 빼고 최종 답만 옮기세요.
JSON으로만 답하세요.
{"class": "반 (없으면 빈 문자열)", "number": "번호", "name": "이름",
 "answers": {"1": {"text": "읽은 글자 그대로 (빈칸이면 빈 문자열)", "messy": true 또는 false, "messy_reason": "지운 자국, 답 두 개, 취소선, 덮어쓰기 등"}, "2": {...}}}"""
MIN_READS, MAX_READS = 2, 3


class NoReading(Exception):
    """미리 읽어 둔 결과도 없고 Gemini 키도 없어 읽을 수 없음"""


class ReadFailed(RuntimeError):
    """다시 시도해도 소용없는 거절. reason은 웹페이지에 보일 짧은 사유, detail은 AI 서비스가 돌려준 원문"""

    def __init__(self, reason, detail=''):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


class ReaderUnavailable(ReadFailed):
    """키, 결제, 모델 문제라 모든 답안지에 똑같이 걸리는 거절. 잠시 호출을 멈춘다"""


# 다시 시도해도 소용없는 응답 코드 → 사유
FATAL_CODES = {401: 'API 키가 올바르지 않음', 402: '결제 크레딧이 없음', 403: '이 키로 모델을 쓸 권한이 없음',
               404: '모델 이름을 찾을 수 없음'}


def classify(error):
    """AI 호출 오류 → (사유, 모든 답안지에 걸리는지). 잠시 뒤 다시 시도하면 될 오류(분당 사용량 제한, 서버 오류,
    응답 시간 초과, 잘린 JSON 등)는 None"""
    code = getattr(error, 'code', None)
    if not isinstance(code, int):
        m = re.match(r'\s*(\d{3})\b', str(error))
        code = int(m.group(1)) if m else None
    msg = str(error)
    if 'API key not valid' in msg or 'API_KEY_INVALID' in msg:   # 잘못된 키는 400으로 온다
        return FATAL_CODES[401], True
    if code == 429 and re.search(r'limit:\s*0\b', msg):          # 이 모델에 무료 한도가 아예 없음
        return '이 키의 등급에서 쓸 수 없는 모델 (무료 한도 0)', True
    if code in FATAL_CODES:
        return FATAL_CODES[code], True
    if code == 400:
        return '요청이 거절됨', False
    return None


class GeminiCall:
    """이미지 한 장과 지시문 → (응답 글자, 토큰 수). 생각 단계 설정을 지원하지 않는 모델이면 끄고 다시 부른다"""

    def __init__(self, api_key, model, temperature=0.7, thinking_level='MINIMAL'):
        from google import genai
        from google.genai import types
        self.types, self.client = types, genai.Client(api_key=api_key)
        self.model, self.temperature, self.thinking_level = model, temperature, thinking_level

    def __call__(self, path, prompt, timeout):
        t = self.types
        kw = dict(temperature=self.temperature, response_mime_type='application/json',
                  http_options=t.HttpOptions(timeout=int(timeout * 1000)))
        if self.thinking_level not in ('', 'OFF'):
            kw['thinking_config'] = t.ThinkingConfig(thinking_level=self.thinking_level)
        with open(path, 'rb') as f:
            image = t.Part.from_bytes(data=f.read(), mime_type='image/png')
        try:
            resp = self.client.models.generate_content(model=self.model, contents=[image, prompt],
                                                       config=t.GenerateContentConfig(**kw))
        except Exception as e:
            if 'thinking' in str(e).lower() and self.thinking_level != 'OFF':
                self.thinking_level = 'OFF'
                return self(path, prompt, timeout)
            raise
        usage = getattr(resp, 'usage_metadata', None)
        return resp.text or '', int(getattr(usage, 'total_token_count', 0) or 0)


class ReadStats:
    """채점 한 번에서 AI 읽기를 몇 번 했는지 (웹페이지의 '읽기' 안내)"""

    def __init__(self):
        self.lock = threading.Lock()
        self.reads = self.new_calls = self.tokens = 0
        self.seconds = []

    def add(self, cached, tokens=0, seconds=0.0):
        with self.lock:
            self.reads += 1
            if not cached:
                self.new_calls += 1
                self.tokens += tokens
                self.seconds.append(seconds)

    def to_dict(self, model):
        return {'reads': self.reads, 'new_calls': self.new_calls, 'cached': self.reads - self.new_calls,
                'tokens': self.tokens, 'model': model,
                'avg_call_seconds': round(sum(self.seconds) / len(self.seconds), 1) if self.seconds else None}


class SheetReader:
    def __init__(self, call, cache_path, model_name=None, call_timeout=45, extra_cache=None, block_seconds=60):
        """call: GeminiCall처럼 (이미지 경로, 지시문, 초) → (글자, 토큰)인 함수. 키가 없으면 None.
        block_seconds: 키, 결제, 모델 문제로 거절되면 이 시간 동안 호출하지 않는다 (그 사이 충전하면 그 뒤에 다시 시도)"""
        self.call, self.model_name, self.call_timeout = call, model_name, call_timeout
        self.block_seconds = block_seconds
        self.blocked = None   # (사유, 원문, 다시 시도할 시각)
        self.cache_path = cache_path
        self.cache = {}
        for p in (extra_cache, cache_path):   # 다른 사람이 미리 읽어 둔 결과 + 이 서버가 읽은 결과
            if p and os.path.exists(p):
                self.cache.update(json.load(open(p, encoding='utf-8')))
        self.lock = threading.Lock()
        self.errors = {}   # 시험지 → 마지막으로 난 읽기 오류 (왜 못 읽었는지 보여 주기 위해)
        self._calls = ThreadPoolExecutor(max_workers=16)   # 호출을 따로 돌려서, 응답이 안 와도 정해진 시간에 반드시 끊음

    @property
    def live(self):
        return self.call is not None

    @property
    def problem(self):
        """키, 결제, 모델 문제로 AI 읽기를 멈춘 상태면 그 사유, 아니면 None"""
        b = self.blocked
        return b[0] if b and time.time() < b[2] else None

    def ask(self, path, prompt, k, sid, stats, deadline=None, aliases=()):
        """이미지 하나를 k번째로 읽힌다 (정답은 알려주지 않음). deadline(시각)까지 못 읽으면 TimeoutError"""
        keys = [f'{os.path.basename(path)}#{k}'] + [f'{a}#{k}' for a in aliases]
        hit = next((key for key in keys if key in self.cache), None)
        if hit:
            stats.add(cached=True)
            return self.cache[hit]
        if self.call is None:
            raise NoReading('미리 읽어 둔 결과가 없고 Gemini 키도 없음')
        b = self.blocked
        if b and time.time() < b[2]:   # 방금 키, 결제 문제로 거절됨 → 부르지 않고 바로 실패
            raise ReaderUnavailable(b[0], b[1])
        last = ''
        for t in range(5):
            left = (deadline - time.time()) if deadline else self.call_timeout
            if left < 2:
                raise TimeoutError('시간 안에 읽지 못함')
            limit = min(self.call_timeout, left)
            t0 = time.time()
            try:
                try:
                    text, tokens = self._calls.submit(self.call, path, prompt, limit).result(timeout=limit + 1)
                except FuturesTimeout:
                    self.errors[sid] = f'{limit:.0f}초 안에 응답 없음 (모델이 느림)'
                    continue
                out = json.loads(re.sub(r'^```(json)?|```$', '', text.strip()).strip())
                if not isinstance(out, dict):
                    raise ValueError('AI 답이 JSON 객체가 아님')
                out['_tokens'] = tokens
                with self.lock:
                    self.cache[keys[0]] = out
                    tmp = self.cache_path + '.tmp'
                    json.dump(self.cache, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False)
                    os.replace(tmp, self.cache_path)
                stats.add(cached=False, tokens=tokens, seconds=time.time() - t0)
                return out
            except Exception as e:
                msg = 'AI 답이 JSON 형식이 아님 (답이 중간에 잘렸을 수 있음)' if isinstance(e, json.JSONDecodeError) else str(e)
                self.errors[sid] = last = msg[:200]
                fatal = classify(e)
                if fatal:   # 다시 시도해도 같은 거절 → 기다리지 않고 바로 실패
                    reason, everywhere = fatal
                    if everywhere:
                        self.blocked = (reason, last, time.time() + self.block_seconds)
                        raise ReaderUnavailable(reason, last) from None
                    raise ReadFailed(reason, last) from None
                wait = 5 * 2 ** t
                m = re.search(r'retry in ([\d.]+)s|retryDelay[^\d]*(\d+)', msg)
                if m:
                    wait = float(m.group(1) or m.group(2)) + 1
                if deadline and time.time() + wait > deadline - 2:   # 기다리면 마감을 넘김 → 이 장은 포기
                    raise TimeoutError(f'시간 안에 읽지 못함 ({msg[:60]})') from None
                time.sleep(min(wait, 30))
        raise ReadFailed('5번 다시 시도해도 읽지 못함', last)

    def read_sheet(self, sid, crops, questions, answer_key, stats, strategy='safe', time_limit=90, aliases=()):
        """한 장을 모아서 읽기 → ((반, 번호, 이름), {문항: 읽기 목록}, {문항: 지저분함})"""
        path = compose_sheet(sid, crops, questions)
        deadline = time.time() + time_limit
        aliases = [f'{a}_sheet.png' for a in aliases]
        get = lambda r, q: (r.get('answers') or {}).get(str(q), {}) or {}
        reads = []
        if strategy == 'fast':   # 한 번 읽고, 애매한 칸이 있을 때만 한 번 더
            for k in range(2):
                try:
                    reads.append(self.ask(path, SHEET_PROMPT, k, sid, stats, deadline, aliases))
                except TimeoutError:
                    if not reads:
                        raise
                    break
                if k == 0 and not any(borderline(answer_key[q], str(get(reads[0], q).get('text', '')),
                                                 bool(get(reads[0], q).get('messy'))) for q in questions):
                    break
            per_q = {q: [str(get(r, q).get('text', '')) for r in reads] for q in questions}
            messy = {q: sum(bool(get(r, q).get('messy')) for r in reads) >= (len(reads) + 1) // 2 for q in questions}
            return _ident(reads[0]), per_q, messy
        # 처음 두 번은 서로의 결과를 기다릴 필요가 없어 동시에 보낸다 (차례로 부를 때보다 시간이 절반 가까이 줄어듦)
        with ThreadPoolExecutor(max_workers=MIN_READS) as ex:
            futures = [ex.submit(self.ask, path, SHEET_PROMPT, k, sid, stats, deadline, aliases) for k in range(MIN_READS)]
        failed = []
        for f in futures:
            try:
                reads.append(f.result())
            except Exception as e:   # 하나만 실패하면 나머지 읽기로 채점 (시간 초과, 일시적 거절 등)
                failed.append(e)
        if not reads:
            raise failed[0]
        texts = [{q: normalize(str(get(r, q).get('text', ''))) for q in questions} for r in reads]
        if MAX_READS > MIN_READS and not failed and not all(len({t[q] for t in texts}) == 1 for q in questions):   # 다르면 세 번째
            try:
                reads.append(self.ask(path, SHEET_PROMPT, MIN_READS, sid, stats, deadline, aliases))
            except TimeoutError:
                pass   # 시간이 모자라면 이미 읽은 결과만 씀
        per_q, messy = {}, {}
        for q in questions:
            items = [get(r, q) for r in reads]
            n = MIN_READS if len({normalize(str(i.get('text', ''))) for i in items[:MIN_READS]}) == 1 else len(items)
            per_q[q] = [str(i.get('text', '')) for i in items[:n]]          # 앞의 두 번이 같은 칸은 두 번만 씀
            messy[q] = sum(bool(i.get('messy')) for i in items[:n]) >= (n + 1) // 2
        return _ident(reads[0]), per_q, messy


def _ident(r):
    return str(r.get('class', '')), str(r.get('number', '')), str(r.get('name', ''))


def borderline(question, text, messy):
    """한 번 더 읽어 볼 만한 칸: 지저분함, 빈칸, 허용 답안과 한두 글자 차이, 헷갈리는 글자만 다름.
    허용 답안과 똑같거나, 등록한 오답이거나, 확실히 다른 답이면 한 번 읽은 걸로 충분"""
    t = normalize(text)
    if messy or not t:
        return True
    if t in question.accepted or t in question.registered_wrong_normalized:
        return False
    return any(edit_distance(t, a) <= 2 or (len(a) == len(t) and all(x == y or is_confusable(x, y) for x, y in zip(t, a)))
               for a in question.accepted)


def compose_sheet(sid, crops, questions):
    """반, 번호, 이름 칸과 문항 칸들을 위에서부터 이어 붙인 이미지 한 장 (각 칸 위에 이름표) → 경로"""
    path = os.path.join(os.path.dirname(crops.flat), f'{sid}_sheet.png')
    if os.path.exists(path):
        return path
    parts = [('반·번호·이름', crops.id)] + \
            [(f'{q}번' + (' (칸 밖 글씨)' if crops.outside[q] else ''), crops.cells[q]) for q in questions]
    W = 768   # 비전 모델이 한 번에 보는 크기에 맞춰 토큰과 시간을 줄임
    tiles = []
    for label, p in parts:
        im = Image.open(p).convert('RGB')
        im = im.resize((W, max(1, int(im.height * W / im.width))))
        tile = Image.new('RGB', (W, im.height + 40), 'white')
        ImageDraw.Draw(tile).text((6, 4), label, font=font(GOTHIC_FONT, 24), fill=(0, 0, 200))
        tile.paste(im, (0, 40))
        tiles.append(tile)
    out = Image.new('RGB', (W, sum(t.height for t in tiles) + 12 * len(tiles)), (200, 200, 200))
    y = 0
    for t in tiles:
        out.paste(t, (0, y))
        y += t.height + 12
    out.save(path)
    return path
