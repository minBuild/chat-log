#!/usr/bin/env bash
# chatlog 한 방 설치 — 새 PC에서 클론/압축해제 후 이 스크립트만 돌리면 된다.
#   ./setup.sh            # 설치 + (있으면) Claude Code 로그 인덱싱 + MCP 등록
#   ./setup.sh --no-mcp   # MCP 등록은 건너뛰기
#   ./setup.sh --no-index # 로그 import/ingest 는 건너뛰기 (코드/환경만)
# 자기 위치를 스스로 찾으므로 폴더를 어디에 두든 동작한다.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

DO_MCP=1; DO_INDEX=1
for arg in "$@"; do
  case "$arg" in
    --no-mcp)   DO_MCP=0 ;;
    --no-index) DO_INDEX=0 ;;
    *) echo "알 수 없는 옵션: $arg"; exit 1 ;;
  esac
done

say() { printf '\n\033[1m▶ %s\033[0m\n' "$1"; }

# 1) Ollama ─────────────────────────────────────────────────────────
say "Ollama 확인"
if ! command -v ollama >/dev/null 2>&1; then
  echo "  ! Ollama 가 없습니다. 먼저 설치하세요:  brew install ollama"
  echo "    설치 후 이 스크립트를 다시 실행하면 됩니다."
  exit 1
fi
# 서버 기동 (이미 떠 있으면 무해)
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "  Ollama 서버 기동 중..."
  brew services start ollama >/dev/null 2>&1 || (ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 1 30); do
    curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi
for m in bge-m3 qwen2.5:7b; do
  if ollama list 2>/dev/null | grep -q "^${m%%:*}"; then
    echo "  ✓ $m 이미 있음"
  else
    echo "  $m 다운로드..."; ollama pull "$m"
  fi
done

# 2) Python venv + 의존성 ───────────────────────────────────────────
say "Python 환경 (.venv)"
PY=python3
command -v "$PY" >/dev/null 2>&1 || { echo "  ! python3 가 필요합니다."; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[mcp]"
echo "  ✓ numpy + mcp 설치 완료"
VENV_PY="$DIR/.venv/bin/python"

# 3) 로그 import + 인덱스 빌드 ──────────────────────────────────────
if [ "$DO_INDEX" -eq 1 ]; then
  say "대화 로그 인덱싱"
  CC_DIR="$HOME/.claude/projects"
  if [ -d "$CC_DIR" ]; then
    echo "  Claude Code 세션 import: $CC_DIR"
    "$VENV_PY" import_logs.py claude-code "$CC_DIR"
  else
    echo "  ! $CC_DIR 없음 — 동봉 examples/ 로 인덱스를 만듭니다(데모)."
    cp -n examples/*.jsonl logs/ 2>/dev/null || true
  fi
  "$VENV_PY" rag.py ingest
else
  echo "  (인덱싱 건너뜀 — 나중에:  .venv/bin/python rag.py ingest)"
fi

# 4) MCP 등록 ───────────────────────────────────────────────────────
if [ "$DO_MCP" -eq 1 ]; then
  say "Claude MCP 등록"
  if command -v claude >/dev/null 2>&1; then
    claude mcp add chatlog "$VENV_PY" "$DIR/mcp_server.py" \
      && echo "  ✓ 등록됨 (claude mcp list 로 확인)"
  else
    echo "  ! 'claude' CLI 가 없어 자동 등록은 건너뜁니다. 아래를 직접 실행/등록하세요:"
    echo
    echo "  claude mcp add chatlog \\"
    echo "    $VENV_PY \\"
    echo "    $DIR/mcp_server.py"
  fi
fi

say "완료 🎉"
echo "  CLI :  .venv/bin/python rag.py ask \"...\""
echo "  웹  :  .venv/bin/python server.py   # http://127.0.0.1:8000"
echo "  MCP :  Claude 세션에서 search_chatlog / ask_chatlog 사용"
