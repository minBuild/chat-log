"""bm25.py + 하이브리드 검색 단위 테스트."""
from conftest import write_log


def test_tokenize_splits_identifiers_and_korean():
    from bm25 import tokenize
    assert tokenize("EasyPaymentController 결제 흐름") == [
        "easypaymentcontroller", "결제", "흐름"]
    assert tokenize("cash_point_balance 적재") == ["cash_point_balance", "적재"]


def test_bm25_ranks_doc_with_exact_term():
    from bm25 import BM25, tokenize
    corpus = [tokenize("캐시 만료 정책"),
              tokenize("filesort 회피 복합 인덱스")]
    bm = BM25(corpus)
    s = bm.scores(tokenize("filesort"))
    assert s[1] > s[0] and s[0] == 0.0


def test_bm25_unknown_term_is_zero():
    from bm25 import BM25, tokenize
    bm = BM25([tokenize("가나다"), tokenize("라마바")])
    assert list(bm.scores(tokenize("zzz없는단어"))) == [0.0, 0.0]


def _seed_for_lexical(rag):
    write_log(rag, "a.jsonl", [{"date": "2025-01-01", "source": "claude",
              "topic": "db", "role": "user", "text": "복합 인덱스 filesort 회피"}])
    write_log(rag, "b.jsonl", [{"date": "2025-01-02", "source": "claude",
              "topic": "cache", "role": "user", "text": "캐시 만료 시간 정책"}])
    rag.cmd_ingest(None)


def test_hybrid_pure_bm25_surfaces_exact_token(rag_env):
    rag = rag_env
    _seed_for_lexical(rag)
    mat, metas = rag.load_index()
    bm25 = rag.build_bm25(metas)
    hits, _ = rag.search("filesort", mat, metas, min_sim=-1.0, alpha=0.0, bm25=bm25)
    assert hits[0][0]["topic"] == "db"          # 정확한 토큰을 가진 문서가 최상위


def test_alpha_one_equals_pure_vector(rag_env):
    rag = rag_env
    _seed_for_lexical(rag)
    mat, metas = rag.load_index()
    bm25 = rag.build_bm25(metas)
    pure, _ = rag.search("캐시 정책", mat, metas, min_sim=0.0, alpha=1.0)
    hybrid_off, _ = rag.search("캐시 정책", mat, metas, min_sim=0.0, alpha=1.0, bm25=bm25)
    # α=1 이면 bm25 를 줘도 무시되고 순수 벡터 결과와 같아야 함
    assert [m["topic"] for m, _ in pure] == [m["topic"] for m, _ in hybrid_off]
