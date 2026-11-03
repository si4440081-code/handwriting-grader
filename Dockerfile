# 독립 시연 서버 이미지. GEMINI_API_KEY는 이미지에 넣지 말고 실행할 때 환경 변수로 넘긴다
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api api
COPY demo demo
COPY grader grader
COPY vision vision
COPY fonts fonts
COPY models models
COPY data data
COPY web web
ENV WORK_DIR=/app/work PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
