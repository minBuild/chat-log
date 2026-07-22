#!/bin/bash
# 컨테이너 시작: Ollama 가 뜰 때까지 대기 → 모델 확보 → 인덱스 없으면 examples 인덱싱 → 서버.
set -euo pipefail

echo "Ollama($OLLAMA_URL) 대기…"
until curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1; do sleep 2; done

# 모델 확보 (이미 있으면 즉시 통과)
for m in "${CHATLOG_EMBED:-bge-m3}" "${CHATLOG_LLM:-qwen2.5:7b}"; do
    echo "모델 확보: $m"
    curl -sf "$OLLAMA_URL/api/pull" -d "{\"model\":\"$m\"}" >/dev/null
done

# 인덱스가 없으면 동봉 예시로 한 번 빌드 (실데이터는 logs/ 를 마운트해서 ingest)
if [ ! -f index.npz ]; then
    echo "인덱스 없음 → examples 로 초기 인덱싱"
    mkdir -p logs
    cp examples/*.jsonl logs/
    python rag.py ingest
fi

echo "서버 시작 → :$PORT"
exec python server.py
