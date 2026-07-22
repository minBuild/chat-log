#!/bin/bash
# 매일 자동 백업(launchd LaunchAgent)을 현재 컴퓨터에 맞게 설치한다.
# plist 템플릿의 경로 placeholder 를 이 프로젝트의 실제 경로로 치환해 등록한다.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="com.chatlog.daily"
AGENTS="$HOME/Library/LaunchAgents"
PLIST="$AGENTS/$LABEL.plist"
UID_NUM="$(id -u)"

# venv 확인 (없으면 안내)
if [ ! -x "$PROJ/.venv/bin/python" ]; then
    echo "먼저 설치가 필요합니다:  python3 -m venv .venv && .venv/bin/pip install numpy" >&2
    exit 1
fi

mkdir -p "$AGENTS"
chmod +x "$PROJ/auto_update.sh"

# 템플릿 → 실제 경로 치환
sed -e "s|__SCRIPT__|$PROJ/auto_update.sh|g" \
    -e "s|__LOG__|$PROJ/auto_update.log|g" \
    "$PROJ/com.chatlog.daily.plist.template" > "$PLIST"

# 기존 등록 해제 후 재등록
launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST"

echo "설치 완료: $LABEL (매일 09:00)"
echo "  즉시 1회 실행 : launchctl kickstart -k gui/$UID_NUM/$LABEL"
echo "  로그 보기      : tail -f $PROJ/auto_update.log"
echo "  제거           : launchctl bootout gui/$UID_NUM/$LABEL && rm $PLIST"
