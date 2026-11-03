"""한글 자모 분해, 조립과 손글씨 혼동쌍 (노트북 1-4)"""

CHO = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')
JUNG = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
JONG = [''] + list('ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ')
# 한글 자모 혼동쌍 (받침 ㄴ/ㅇ, ㄴ/ㅁ 포함)
CONFUSABLE = [('ㅁ', 'ㅂ'), ('ㅐ', 'ㅔ'), ('ㅗ', 'ㅜ'), ('ㅏ', 'ㅓ'), ('ㅅ', 'ㅈ'), ('ㄴ', 'ㄹ'), ('ㅇ', 'ㅁ'),
              ('ㅓ', 'ㅕ'), ('ㅏ', 'ㅑ'), ('ㅗ', 'ㅛ'), ('ㅜ', 'ㅠ'), ('ㅡ', 'ㅜ'), ('ㄱ', 'ㅋ'), ('ㄷ', 'ㅌ'),
              ('ㄴ', 'ㅇ'), ('ㄴ', 'ㅁ')]
# 숫자와 영어 소문자 혼동 (글자 → 잘못 읽힐 수 있는 글자들)
SIMPLE_CONFUSABLE = {'1': ['7', 'l'], '7': ['1'], '3': ['8'], '8': ['3'], '0': ['6', 'o'], '6': ['0'],
                     'o': ['0', 'a'], 'a': ['o'], 'l': ['1', 'i'], 'i': ['l'], 'u': ['v'], 'v': ['u'],
                     'n': ['h'], 'h': ['n'], 'e': ['c'], 'c': ['e']}


def decompose(ch):
    """완성형 한글 한 글자 → (초성, 중성, 종성). 한글이 아니면 None"""
    code = ord(ch) - 0xAC00
    if not (0 <= code < 11172):
        return None
    return CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]


def compose(cho, jung, jong=''):
    return chr(0xAC00 + (CHO.index(cho) * 21 + JUNG.index(jung)) * 28 + JONG.index(jong))


def is_confusable(a, b):
    """두 글자가 숫자, 영어 혼동이거나 자모 하나만 혼동쌍(또는 받침 있음/없음)으로 다르면 True"""
    if b in SIMPLE_CONFUSABLE.get(a, []) or a in SIMPLE_CONFUSABLE.get(b, []):
        return True
    da, db = decompose(a), decompose(b)
    if da is None or db is None:
        return False
    diffs = [(x, y) for x, y in zip(da, db) if x != y]
    if len(diffs) != 1:
        return False
    x, y = diffs[0]
    return x == '' or y == '' or (x, y) in CONFUSABLE or (y, x) in CONFUSABLE
