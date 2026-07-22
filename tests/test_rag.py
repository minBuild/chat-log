"""rag.py 단위 테스트 — 청킹, 인덱싱, 검색, 필터, 임계값."""
import os

from conftest import write_log


# ── 청킹 ───────────────────────────────────────────────────────────
def test_chunk_text_covers_and_overlaps():
    import rag
    text = "가" * 2000
    chunks = rag.chunk_text(text)
    # 모든 청크를 이으면(겹침 제거) 원본을 빠짐없이 덮어야 한다.
    assert "".join(c[: rag.CHUNK_SIZE - rag.CHUNK_OVERLAP] for c in chunks).startswith("가" * 100)
    assert all(len(c) <= rag.CHUNK_SIZE for c in chunks)
    # 인접 청크는 CHUNK_OVERLAP 만큼 겹친다.
    assert chunks[0][-rag.CHUNK_OVERLAP:] == chunks[1][: rag.CHUNK_OVERLAP]


def test_short_text_single_chunk():
    import rag
    assert rag.chunk_text("짧은 글") == ["짧은 글"]


def test_chunk_records_keeps_message_boundaries():
    import rag
    recs = [{"role": "user", "text": "가" * 300},
            {"role": "assistant", "text": "나" * 300},
            {"role": "user", "text": "다" * 300}]
    chunks = rag.chunk_records(recs)
    # 한도(800) 안에서 메시지를 통째로 묶고, 메시지를 쪼개지 않는다.
    assert all(len(c) <= rag.CHUNK_SIZE for c in chunks)
    assert len(chunks) >= 2
    # 각 줄은 "role: text" 형태로 온전히 보존 (중간 절단 없음)
    for c in chunks:
        for line in c.split("\n"):
            assert line.startswith(("user:", "assistant:"))


def test_chunk_records_overlaps_one_message():
    import rag
    recs = [{"role": "user", "text": "가" * 500},
            {"role": "assistant", "text": "나" * 500}]
    chunks = rag.chunk_records(recs)
    assert len(chunks) == 2
    # 두 번째 청크는 직전 메시지를 겹쳐 시작한다(문맥 유지).
    assert chunks[1].startswith("user: " + "가" * 10)


def test_chunk_records_splits_oversized_message():
    import rag
    recs = [{"role": "user", "text": "가" * 2000}]
    chunks = rag.chunk_records(recs)
    assert len(chunks) > 1 and all(len(c) <= rag.CHUNK_SIZE for c in chunks)


def test_hash_is_deterministic():
    import rag
    assert rag._hash("같은 글") == rag._hash("같은 글")
    assert rag._hash("a") != rag._hash("b")


# ── 로딩 ───────────────────────────────────────────────────────────
def test_load_records_skips_bad_json(rag_env, capsys):
    rag = rag_env
    path = os.path.join(rag.LOG_DIR, "x.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"text":"좋음"}\n')
        f.write("깨진 줄\n")
        f.write('{"text":"또좋음"}\n')
    recs = rag.load_records(path)
    assert [r["text"] for r in recs] == ["좋음", "또좋음"]


# ── 인덱싱 + 검색 ──────────────────────────────────────────────────
def _seed(rag):
    write_log(rag, "2025-01-15_caching.jsonl", [
        {"date": "2025-01-15", "source": "claude", "topic": "caching",
         "role": "user", "text": "캐시 ttl 만료 시간 정책"},
    ])
    write_log(rag, "2025-03-22_retry.jsonl", [
        {"date": "2025-03-22", "source": "claude", "topic": "retry",
         "role": "user", "text": "재시도 지수 백오프 지터"},
    ])
    rag.cmd_ingest(None)


def test_ingest_writes_index(rag_env):
    rag = rag_env
    _seed(rag)
    assert os.path.exists(rag.INDEX_NPZ)
    mat, metas = rag.load_index()
    assert mat.shape[0] == len(metas) == 2
    assert {m["topic"] for m in metas} == {"caching", "retry"}


def test_search_ranks_relevant_first(rag_env):
    rag = rag_env
    _seed(rag)
    mat, metas = rag.load_index()
    hits, _ = rag.search("캐시 만료 ttl 어떻게", mat, metas, min_sim=0.0)
    assert hits[0][0]["topic"] == "caching"


def test_topic_filter(rag_env):
    rag = rag_env
    _seed(rag)
    mat, metas = rag.load_index()
    hits, _ = rag.search("재시도 백오프", mat, metas, min_sim=0.0, topic="retry")
    assert hits and all(m["topic"] == "retry" for m, _ in hits)


def test_topic_filter_no_match_returns_none(rag_env):
    rag = rag_env
    _seed(rag)
    mat, metas = rag.load_index()
    hits, max_sim = rag.search("아무거나", mat, metas, topic="없는주제")
    assert hits is None and max_sim is None


def test_min_sim_threshold_drops_irrelevant(rag_env):
    rag = rag_env
    _seed(rag)
    mat, metas = rag.load_index()
    # 공유 단어가 전혀 없는 질문 → 유사도 0 → 임계값에 걸려 결과 없음
    hits, max_sim = rag.search("전혀 관계없는 외계어 zzzz", mat, metas, min_sim=0.45)
    assert hits == []
    assert max_sim < 0.45


def test_date_filter(rag_env):
    rag = rag_env
    _seed(rag)
    mat, metas = rag.load_index()
    hits, _ = rag.search("정책 백오프", mat, metas, min_sim=0.0, since="2025-03-01")
    assert hits and all(m["date"] >= "2025-03-01" for m, _ in hits)


def test_ingest_reuses_unchanged_embeddings(rag_env):
    rag = rag_env
    _seed(rag)
    calls = {"n": 0}
    orig = rag.embed

    def counting(text):
        calls["n"] += 1
        return orig(text)

    rag.embed = counting
    rag.cmd_ingest(None)              # 내용 동일 → 전부 캐시 재사용
    assert calls["n"] == 0, "변경 없는 청크는 재임베딩하면 안 된다(증분)"
