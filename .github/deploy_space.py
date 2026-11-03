"""GitHub Actions에서 실행: 이 저장소를 Hugging Face Space(Docker)로 올리고, 키와 설정을 Space에 넣는다."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

token, space = os.environ.get('HF_TOKEN'), os.environ.get('HF_SPACE')
if not token or not space:
    sys.exit('HF_TOKEN(Secrets)과 HF_SPACE(Variables)를 GitHub 저장소 설정에 넣어 주세요')
api = HfApi(token=token)
api.create_repo(space, repo_type='space', space_sdk='docker', exist_ok=True, private=False)

for name in ('GEMINI_API_KEY', 'DEMO_ACCESS_CODE'):
    if os.environ.get(name):
        api.add_space_secret(space, name, os.environ[name])
        print(f'Secret 설정: {name}')
for name in ('GEMINI_MODEL', 'DEMO_DAILY_LIMIT', 'DEFAULT_SUBJECT', 'READ_WORKERS'):
    if os.environ.get(name):
        api.add_space_variable(space, name, os.environ[name])
        print(f'Variable 설정: {name} = {os.environ[name]}')

root = Path(__file__).resolve().parent.parent
keep = ['api', 'demo', 'grader', 'vision', 'fonts', 'models', 'data', 'web', 'Dockerfile', 'requirements.txt', '.dockerignore']
with tempfile.TemporaryDirectory() as tmp:
    for item in keep:
        src = root / item
        if src.is_dir():
            shutil.copytree(src, Path(tmp) / item, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        elif src.exists():
            shutil.copy(src, Path(tmp) / item)
    header = ('---\ntitle: 손글씨 채점 AI\nemoji: ✏️\ncolorFrom: red\ncolorTo: blue\nsdk: docker\napp_port: 8000\npinned: false\n---\n\n')
    (Path(tmp) / 'README.md').write_text(header + (root / 'README.md').read_text(encoding='utf-8'), encoding='utf-8')
    api.upload_folder(folder_path=tmp, repo_id=space, repo_type='space', commit_message='GitHub에서 자동 배포',
                      delete_patterns=['*'])
print(f'완료: https://huggingface.co/spaces/{space}  (빌드에 몇 분 걸림)')
