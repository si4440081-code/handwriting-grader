# GitHub에 올리고 시연 서버 배포하기

키는 **파일에 절대 넣지 않고** GitHub 저장소의 Secrets에만 넣습니다.

## 1. 저장소 만들고 올리기
1. github.com → New repository → 이름(예: handwriting-grader) → Public → Create
2. "uploading an existing file" → 이 폴더 안의 파일과 폴더를 모두 끌어다 놓기 (`.github` 폴더 포함) → Commit
   - 올리기 전에 `.env`, `work/`, 키가 적힌 파일이 없는지 확인 (`.gitignore`가 막아 줌)

## 2. Secrets와 Variables 넣기
저장소 → Settings → Secrets and variables → Actions

| 탭 | 이름 | 값 |
|---|---|---|
| Secrets | `GEMINI_API_KEY` | AI Studio에서 만든 키 |
| Secrets | `DEMO_ACCESS_CODE` | 시연용 접근 코드 (원하는 값) |
| Secrets | `HF_TOKEN` | Hugging Face → Settings → Access Tokens → **Write** 토큰 |
| Variables | `HF_SPACE` | `내HF아이디/handwriting-grader` |
| Variables | `GEMINI_MODEL` | 예: `gemini-3.8-flash` |
| Variables | `DEMO_DAILY_LIMIT` | `40` |

## 3. 배포
- Actions 탭 → "시연 서버 배포 (Hugging Face)" → Run workflow (이후로는 main에 올릴 때마다 자동)
- 몇 분 뒤 `https://huggingface.co/spaces/내아이디/handwriting-grader` 가 시연 주소
- 처음 켜질 때 예시 답안지를 미리 읽어서 1~2분 걸릴 수 있음. 한동안 접속이 없으면 잠들었다가 첫 접속 때 다시 깨어남

## 4. 테스트
- 올릴 때마다 Actions의 "테스트"가 자동으로 돌아감 (키 없이 가짜 읽기로 확인)
