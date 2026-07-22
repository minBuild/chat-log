#!/bin/bash
# chatlog 자동 갱신: Claude Code 세션 로그 수집 + 인덱스 증분 갱신
# launchd(LaunchAgent)가 하루 한 번 실행한다.
set -uo pipefail

# 스크립트 자신의 위치를 기준으로 프로젝트 경로를 잡는다(어느 컴퓨터든 이식 가능).
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$PROJ/.venv/bin/python"
cd "$PROJ" || exit 1

echo "================ $(date '+%Y-%m-%d %H:%M:%S') ================"

# 1) Claude Code 세션 transcript → logs/ (세션당 파일 1개, 멱등 덮어쓰기)
"$PY" import_logs.py claude-code "$HOME/.claude/projects"

# 2) 인덱스 증분 갱신 (바뀐/새 청크만 임베딩)
"$PY" rag.py ingest

echo "완료: $(date '+%H:%M:%S')"
