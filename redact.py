"""
redact : 대화 로그·메모에 섞여 들어온 비밀값(토큰·키·비밀번호·복구 코드)을 마스킹한다.

AI 대화에는 Notion/Confluence/코드에서 복사된 토큰이 그대로 실리기 쉽다. 로그를 로컬에만 두더라도
인덱스 → 검색 결과 → 다음 대화로 다시 번지므로, import 와 결론 메모 저장 시점에 한 번 지운다.

패턴은 정밀도 우선이다(오탐이 적은 것만). 모양이 뚜렷한 토큰은 통째로 가리고,
`api_key=값` 꼴은 키 이름을 남긴 채 값만 가린다. 마스킹된 자리는 `[REDACTED:종류]` 로 남아
"여기에 뭔가 있었다"는 사실은 검색된다.
"""
from __future__ import annotations

import re
from collections.abc import Callable

MASK = "[REDACTED:{kind}]"

# (종류, 정규식). 그룹이 2개면 1번 그룹은 남기고 2번 그룹만 가린다(예: "Bearer " 접두사 유지).
_TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private-key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("github", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("slack", re.compile(r"\bxox[abeprs]-[A-Za-z0-9-]{10,}\b")),
    ("aws", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("openai", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("bearer", re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._~+/=-]{20,})")),
]

# key: 값 / key=값 꼴 — 키 이름(1)과 구분자(2)는 남기고 값(3)만 가린다.
# 키 이름은 api_key·secret·token·password 로 끝나는 식별자(GITHUB_TOKEN, ELEVEN_API_KEY 등).
# 값이 8자 미만이면 비밀로 보지 않고, 이미 가려진 값([REDACTED:…])은 다시 세지 않는다.
_ASSIGN_RE = re.compile(
    r"(?i)\b([A-Za-z0-9_.-]*?(?:api[_-]?key|secret[_-]?key|client[_-]?secret|access[_-]?token|"
    r"refresh[_-]?token|auth[_-]?token|private[_-]?key|password|passwd|secret|token))\b"
    r"(\s*[:=]\s*[\"']?)(?!\[REDACTED)([^\s\"',;]{8,})"
)

# "recovery code" / "복구 코드" 가 적힌 줄에서는 코드 모양 토큰을 전부 가린다
# (abcde-12345 처럼 하이픈으로 이어진 것, 숫자가 섞인 8자 이상 영숫자).
_RECOVERY_LINE_RE = re.compile(r"(?i)recovery\s*codes?|복구\s*코드|백업\s*코드")
_CODE_RE = re.compile(r"\b[A-Za-z0-9]{4,}-[A-Za-z0-9]{4,}\b|\b(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{8,}\b")

Repl = Callable[[re.Match[str]], str]


def _token_repl(mask: str) -> Repl:
    def repl(m: re.Match[str]) -> str:
        return (m.group(1) if m.re.groups >= 2 else "") + mask
    return repl


def _assign_repl(m: re.Match[str]) -> str:
    return m.group(1) + m.group(2) + MASK.format(kind="secret")


def redact_text(text: str) -> tuple[str, int]:
    """비밀값을 마스킹한 텍스트와 마스킹 건수를 돌려준다. 비밀값이 없으면 원문 그대로, 0."""
    count = 0

    def sub(pattern: re.Pattern[str], repl: Repl | str, s: str) -> str:
        nonlocal count
        s, n = pattern.subn(repl, s)
        count += n
        return s

    # 1) 모양이 뚜렷한 토큰 — 종류까지 표시한다
    for kind, pat in _TOKEN_PATTERNS:
        text = sub(pat, _token_repl(MASK.format(kind=kind)), text)
    # 2) key=값 꼴의 나머지
    text = sub(_ASSIGN_RE, _assign_repl, text)
    # 3) 복구 코드 줄
    if _RECOVERY_LINE_RE.search(text):
        lines = []
        for line in text.split("\n"):
            if _RECOVERY_LINE_RE.search(line):
                line = sub(_CODE_RE, MASK.format(kind="recovery-code"), line)
            lines.append(line)
        text = "\n".join(lines)
    return text, count
