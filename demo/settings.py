"""시연 서버 설정 (환경 변수로 바꾼다. 기본값은 노트북 1-2, 5-2와 같다)"""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name, default, cast=str):
    value = os.environ.get(name)
    if value is None or value == '':
        return default
    if cast is bool:
        return value.lower() in ('1', 'true', 'yes', 'on')
    return cast(value)


@dataclass
class Settings:
    data_dir: Path = ROOT / 'data'            # subjects/과목/ 정답표, 명렬표, sheets/ 예시 답안지
    work_dir: Path = ROOT / 'work'            # 펼친 답안지, 자른 칸, 채점 결과, AI 읽기 저장
    web_page: Path = ROOT / 'web' / 'index.html'
    default_subject: str = '역사'             # 웹페이지에서 처음 고른 과목 (정답표를 올리지 않으면 이 과목)
    gemini_api_key: str = field(default='', repr=False)
    gemini_model: str = 'gemini-3.8-flash'
    sample_read_strategy: str = 'safe'        # 예시 답안지는 정확도 우선 (미리 읽어 둠)
    sample_time_limit: int = 90
    upload_read_strategy: str = 'fast'        # 웹페이지에서 새로 올린 사진은 빠르게
    upload_time_limit: int = 30
    read_workers: int = 4                     # 동시에 읽을 답안지 수 (분당 사용량 제한에 걸리면 줄이기)
    prefetch: bool = True                     # 서버를 켤 때 예시 답안지를 미리 읽어 둠
    access_code: str = field(default='', repr=False)   # 정하면 '내 답안지 올리기'에 이 코드가 필요
    daily_limit: int = 40                     # 하루에 새로 읽는 답안지 수 상한 (키 사용량 보호)
    max_upload_mb: int = 50
    n_samples: int = 3000                     # 시뮬레이션 학습 데이터 개수
    seed: int = 42

    @classmethod
    def from_env(cls):
        d = cls()
        return cls(
            data_dir=Path(_env('DATA_DIR', d.data_dir)), work_dir=Path(_env('WORK_DIR', d.work_dir)),
            web_page=Path(_env('WEB_PAGE', d.web_page)), default_subject=_env('DEFAULT_SUBJECT', d.default_subject),
            gemini_api_key=_env('GEMINI_API_KEY', ''), gemini_model=_env('GEMINI_MODEL', d.gemini_model),
            sample_read_strategy=_env('SAMPLE_READ_STRATEGY', d.sample_read_strategy),
            sample_time_limit=_env('SAMPLE_TIME_LIMIT', d.sample_time_limit, int),
            upload_read_strategy=_env('UPLOAD_READ_STRATEGY', d.upload_read_strategy),
            upload_time_limit=_env('UPLOAD_TIME_LIMIT', d.upload_time_limit, int),
            read_workers=_env('READ_WORKERS', d.read_workers, int), prefetch=_env('PREFETCH', d.prefetch, bool),
            access_code=_env('DEMO_ACCESS_CODE', ''), daily_limit=_env('DEMO_DAILY_LIMIT', d.daily_limit, int),
            max_upload_mb=_env('MAX_UPLOAD_MB', d.max_upload_mb, int),
            n_samples=_env('N_SAMPLES', d.n_samples, int), seed=_env('SEED', d.seed, int),
        )
