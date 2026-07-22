"""import_logs.py 단위 테스트 — 변환 정확도와 멱등성."""
import json
import os


def test_slugify_keeps_korean_drops_symbols():
    import import_logs as il
    assert il.slugify("결제 / API: 검토!") == "결제-API-검토"
    assert il.slugify("") == "untitled"


def test_assistant_text_extracts_only_text_blocks():
    import import_logs as il
    content = [
        {"type": "thinking", "thinking": "속으로 생각"},
        {"type": "text", "text": "보이는 답"},
        {"type": "tool_use", "name": "Bash"},
    ]
    assert il._cc_assistant_text(content) == "보이는 답"
    assert il._cc_assistant_text("문자열 그대로") == "문자열 그대로"


def _write_session(path):
    """Claude Code 세션 transcript 한 개를 흉내낸다."""
    lines = [
        {"type": "mode", "mode": "normal"},
        {"type": "ai-title", "aiTitle": "결제 흐름 분석"},
        {"type": "user", "timestamp": "2025-05-01T09:00:00Z", "cwd": "/Users/x/payflow",
         "message": {"role": "user", "content": "결제 흐름 알려줘"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "생각중"},
            {"type": "text", "text": "이렇게 흐릅니다"}]}},
        # tool_result(user-list) 는 노이즈 → 버려져야 함
        {"type": "user", "message": {"role": "user",
         "content": [{"type": "tool_result", "content": "ls 결과..."}]}},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for o in lines:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def test_parse_session_extracts_dialog_and_topic(tmp_path):
    import import_logs as il
    sess = tmp_path / "abcd1234.jsonl"
    _write_session(str(sess))
    title, topic, recs = il._cc_parse_session(str(sess))
    assert title == "결제 흐름 분석"
    assert topic == "payflow"                       # cwd 의 마지막 경로 조각
    assert [(r["role"], r["text"]) for r in recs] == [
        ("user", "결제 흐름 알려줘"),
        ("assistant", "이렇게 흐릅니다"),
    ]                                               # tool_result 는 제외됨


def test_claude_code_import_is_idempotent(tmp_path, monkeypatch):
    import import_logs as il
    src = tmp_path / "src"
    src.mkdir()
    _write_session(str(src / "session-uuid-1.jsonl"))
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(il, "LOG_DIR", str(logs))

    il.import_claude_code(str(src), None)
    first = sorted(os.listdir(logs))
    il.import_claude_code(str(src), None)           # 한 번 더
    second = sorted(os.listdir(logs))
    assert first == second and len(first) == 1, "재실행해도 파일이 늘면 안 된다(멱등)"


def test_claude_code_prune_removes_stale_sessions(tmp_path, monkeypatch):
    import import_logs as il
    src = tmp_path / "src"
    src.mkdir()
    # 세션 파일명이 uuid 형태여야 prune 패턴에 매칭된다
    _write_session(str(src / "aaaaaaaa-1111-2222-3333-444444444444.jsonl"))
    _write_session(str(src / "bbbbbbbb-5555-6666-7777-888888888888.jsonl"))
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(il, "LOG_DIR", str(logs))

    il.import_claude_code(str(src), None)
    assert len(list(logs.glob("*.jsonl"))) == 2

    (src / "bbbbbbbb-5555-6666-7777-888888888888.jsonl").unlink()   # 원본 세션 삭제
    il.import_claude_code(str(src), None, prune=True)
    remaining = list(logs.glob("*.jsonl"))
    assert len(remaining) == 1 and "aaaaaaaa" in remaining[0].name


def test_claude_export_import(tmp_path, monkeypatch):
    import import_logs as il
    conv = tmp_path / "conversations.json"
    conv.write_text(json.dumps([{
        "uuid": "c1", "name": "캐시 정책", "created_at": "2025-01-15T00:00:00Z",
        "chat_messages": [
            {"sender": "human", "created_at": "2025-01-15T00:00:00Z", "text": "왜 24시간?"},
            {"sender": "assistant", "created_at": "2025-01-15T00:01:00Z", "text": "하루 갱신이라"},
        ],
    }], ensure_ascii=False), encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(il, "LOG_DIR", str(logs))

    il.import_claude_export(str(conv), topic_override="caching")
    out = list(logs.glob("*.jsonl"))
    assert len(out) == 1
    recs = [json.loads(line) for line in out[0].read_text(encoding="utf-8").splitlines()]
    assert [r["role"] for r in recs] == ["user", "assistant"]
    assert all(r["topic"] == "caching" for r in recs)
    assert recs[0]["date"] == "2025-01-15"
