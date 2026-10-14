"""notes.py 단위 테스트 — 결론 메모 저장·조회·마스킹, 인덱스 포함, 검색 우선순위, 중복 제거."""
import json
import os

import pytest
from conftest import write_log


def test_add_note_appends_record(notes_env):
    notes = notes_env
    rec = notes.add_note("11번가 키 발급", "셀러오피스 로그인 뒤 API 관리에서 받는다",
                         sources=["https://openapi.11st.co.kr", " PAY-1 "], topic="lowpick",
                         tags=["api", ""], date="2026-10-13")
    assert rec["id"] and rec["date"] == "2026-10-13" and rec["topic"] == "lowpick"
    assert rec["sources"] == ["https://openapi.11st.co.kr", "PAY-1"]     # 공백·빈 값 정리
    assert rec["tags"] == ["api"]
    notes.add_note("두 번째", "본문")                                       # 기본 topic 은 note
    with open(notes.NOTES_FILE, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert [n["title"] for n in lines] == ["11번가 키 발급", "두 번째"]      # append-only
    assert lines[1]["topic"] == "note"


def test_add_note_rejects_empty(notes_env):
    with pytest.raises(ValueError):
        notes_env.add_note("  ", "본문")
    with pytest.raises(ValueError):
        notes_env.add_note("제목", "")


def test_add_note_redacts_secrets(notes_env):
    rec = notes_env.add_note("토큰 메모", "발급 토큰 ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 저장",
                             sources=["xoxb-1234567890-abcdefghij"])
    assert "ghp_" not in rec["text"] and "[REDACTED:github]" in rec["text"]
    assert rec["sources"] == ["[REDACTED:slack]"]
    assert rec["redacted"] == 1                           # 본문 기준 건수(출처는 별도 마스킹)


def test_list_notes_newest_first_with_filters(notes_env):
    notes = notes_env
    notes.add_note("a", "1", topic="x", date="2026-01-01")
    notes.add_note("b", "2", topic="y", date="2026-02-01")
    notes.add_note("c", "3", topic="x", date="2026-03-01")
    assert [n["title"] for n in notes.list_notes()] == ["c", "b", "a"]
    assert [n["title"] for n in notes.list_notes(topic="x")] == ["c", "a"]
    assert [n["title"] for n in notes.list_notes(since="2026-02-01")] == ["c", "b"]
    assert [n["title"] for n in notes.list_notes(limit=1)] == ["c"]


def test_load_notes_skips_bad_lines_and_missing_file(notes_env):
    notes = notes_env
    assert notes.load_notes() == []
    os.makedirs(os.path.dirname(notes.NOTES_FILE))
    with open(notes.NOTES_FILE, "w", encoding="utf-8") as f:
        f.write('{"title":"ok","text":"t"}\n깨진 줄\n')
    assert [n["title"] for n in notes.load_notes()] == ["ok"]


def test_iter_note_chunks_one_chunk_per_note(notes_env):
    notes = notes_env
    notes.add_note("캐시 TTL", "24시간으로 결정", sources=["https://x/wiki"], tags=["결정"],
                   topic="caching")
    chunks = list(notes.iter_note_chunks())
    assert len(chunks) == 1
    text, meta = chunks[0]
    assert text.startswith("결론 메모: 캐시 TTL\n24시간으로 결정\n출처: https://x/wiki")
    assert meta["source"] == "note" and meta["topic"] == "caching" and meta["file"] == "notes.jsonl"
    assert meta["note_id"] and meta["title"] == "캐시 TTL"


def test_to_markdown(notes_env):
    notes = notes_env
    notes.add_note("제목", "본문", sources=["S1"], date="2026-01-01")
    md = notes.to_markdown(notes.load_notes())
    assert md.startswith("# 결론 메모") and "## 2026-01-01 · 제목" in md and "- 출처: S1" in md


# ── rag 와의 결합 ──────────────────────────────────────────────────
def test_ingest_includes_notes_and_search_puts_notes_first(rag_env, notes_env):
    rag, notes = rag_env, notes_env
    write_log(rag, "2025-01-15_caching.jsonl", [
        {"date": "2025-01-15", "source": "claude", "topic": "caching",
         "role": "user", "text": "캐시 ttl 정책"},            # 질문과 같은 단어 → 로그가 더 유사
    ])
    notes.add_note("캐시 ttl", "정책은 24시간", topic="caching", date="2025-01-16")
    rag.cmd_ingest(None)
    mat, metas = rag.load_index()
    assert {m["source"] for m in metas} == {"claude", "note"}

    hits, _ = rag.search("캐시 ttl 정책", mat, metas, min_sim=0.0, alpha=1.0)
    assert hits[0][0]["source"] == "note"                      # 메모가 먼저
    assert hits[1][0]["source"] == "claude"
    hits, _ = rag.search("캐시 ttl 정책", mat, metas, min_sim=0.0, alpha=1.0, notes_first=False)
    assert hits[0][0]["source"] == "claude"                    # 끄면 점수순(로그가 더 유사)


def test_note_below_threshold_is_not_forced_in(rag_env, notes_env):
    rag, notes = rag_env, notes_env
    write_log(rag, "2025-03-22_retry.jsonl", [
        {"date": "2025-03-22", "source": "claude", "topic": "retry",
         "role": "user", "text": "재시도 지수 백오프 지터"},
    ])
    notes.add_note("전혀 다른 주제", "배포 롤백 절차", topic="deploy")
    rag.cmd_ingest(None)
    mat, metas = rag.load_index()
    hits, _ = rag.search("재시도 백오프", mat, metas, min_sim=0.3, alpha=1.0)
    assert [m["source"] for m, _ in hits] == ["claude"]        # 무관한 메모는 임계값에 걸려 빠진다


def test_prompt_marks_notes(rag_env):
    rag = rag_env
    hits = [({"date": "d", "source": "note", "topic": "t", "text": "메모"}, 0.9),
            ({"date": "d", "source": "claude", "topic": "t", "text": "로그"}, 0.8)]
    prompt = rag.build_prompt("질문", hits)
    assert "★ 결론 메모 [d · note · t]" in prompt and "[d · claude · t]" in prompt
    assert "메모를 우선" in prompt


def test_ingest_drops_duplicate_chunks(rag_env, capsys):
    rag = rag_env
    rec = {"date": "2025-01-15", "source": "claude", "topic": "caching", "role": "user",
           "text": "같은 페이지 본문이 두 세션에 반복"}
    write_log(rag, "2025-01-15_a.jsonl", [rec])
    write_log(rag, "2025-01-16_b.jsonl", [rec])
    rag.cmd_ingest(None)
    mat, metas = rag.load_index()
    assert len(metas) == 1 and mat.shape[0] == 1
    assert "중복 제거 1" in capsys.readouterr().out
