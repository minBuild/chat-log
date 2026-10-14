"""redact.py 단위 테스트 — 비밀값 마스킹(정밀도 우선: 일반 문장은 건드리지 않는다)."""
from redact import redact_text


def test_plain_text_untouched():
    text = "캐시 TTL 은 24시간으로 잡았다. 2026-10-13 에 결정. PAY-123 참고."
    assert redact_text(text) == (text, 0)


def test_masks_known_token_shapes():
    text = ("github ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 와 slack xoxb-1234567890-abcdefghij "
            "aws AKIAIOSFODNN7EXAMPLE "
            "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.SflKxwRJSMeKKF2QT4fwpM")
    out, n = redact_text(text)
    assert n == 4
    assert "[REDACTED:github]" in out and "[REDACTED:slack]" in out
    assert "[REDACTED:aws]" in out and "[REDACTED:jwt]" in out
    assert "ghp_" not in out and "xoxb-" not in out and "AKIA" not in out


def test_bearer_keeps_prefix_and_private_key_block():
    out, n = redact_text("Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123\n"
                         "-----BEGIN PRIVATE KEY-----\nMIIE...\n-----END PRIVATE KEY-----")
    assert n == 2
    assert "Bearer [REDACTED:bearer]" in out
    assert "MIIE" not in out and "[REDACTED:private-key]" in out


def test_assignment_keeps_key_name_masks_value_once():
    out, n = redact_text("GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
                         "password: hunter2secret ELEVEN_API_KEY=0123456789abcdef")
    assert n == 3
    assert out == ("GITHUB_TOKEN=[REDACTED:github] password: [REDACTED:secret] "
                   "ELEVEN_API_KEY=[REDACTED:secret]")     # 토큰 모양은 종류까지, 나머지는 값만


def test_short_values_are_not_secrets():
    text = "token: abc 비밀번호는 바꿨다"
    assert redact_text(text) == (text, 0)


def test_recovery_code_line_only():
    text = "복구 코드: a1b2c-3d4e5 f6g7h-8i9j0\n다음 회의는 2026-10-20 10:00"
    out, n = redact_text(text)
    assert n == 2
    assert out.startswith("복구 코드: [REDACTED:recovery-code] [REDACTED:recovery-code]")
    assert out.endswith("다음 회의는 2026-10-20 10:00")     # 날짜·시간은 그대로
