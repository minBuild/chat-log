FROM python:3.12-slim

WORKDIR /app

# 의존성 먼저 (레이어 캐시)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[web]"

# 소스
COPY rag.py bm25.py import_logs.py server.py docker-entrypoint.sh ./
COPY web ./web
COPY examples ./examples
RUN chmod +x docker-entrypoint.sh

ENV OLLAMA_URL=http://ollama:11434 HOST=0.0.0.0 PORT=8000
EXPOSE 8000

# 첫 실행 시 모델 받고 examples 인덱싱 후 서버 기동
ENTRYPOINT ["./docker-entrypoint.sh"]
