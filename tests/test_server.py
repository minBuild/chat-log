"""FastAPI 엔드포인트 테스트 — rag 검색/생성을 가짜로 주입해 Ollama 없이 검증.

fastapi/httpx 가 없으면(코어만 설치한 환경) 자동 skip."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def rag_mod(monkeypatch):
    import numpy as np

    import rag

    metas = [{"date": "2025-01-15", "file": "c.jsonl", "topic": "caching",
              "text": "캐시 ttl 24시간 정책", "source": "claude", "chunk": 0}]
    monkeypatch.setattr(rag, "load_index", lambda: (np.zeros((1, 4), dtype="float32"), metas))
    monkeypatch.setattr(rag, "build_bm25", lambda m: None)
    monkeypatch.setattr(rag, "search", lambda *a, **k: ([(metas[0], 0.72)], 0.72))
    monkeypatch.setattr(rag, "generate", lambda prompt: "24시간으로 잡았습니다.")
    monkeypatch.setattr(rag, "generate_stream", lambda prompt: iter(["24시간", "으로."]))
    return rag


@pytest.fixture
def client(rag_mod):
    import server
    with TestClient(server.app) as c:      # with-블록이 lifespan(startup) 을 실행
        yield c


def test_ask_returns_answer_and_sources(client):
    r = client.post("/api/ask", json={"question": "캐시 왜 24시간?"})
    assert r.status_code == 200
    d = r.json()
    assert d["answer"] == "24시간으로 잡았습니다."
    assert d["sources"][0]["topic"] == "caching"
    assert d["sources"][0]["score"] == 0.72


def test_ask_generate_false_skips_llm(client):
    r = client.post("/api/ask", json={"question": "캐시", "generate": False})
    d = r.json()
    assert d["answer"] is None
    assert d["sources"]                         # 검색 결과는 그대로 온다


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>" in r.text


def test_health(client):
    d = client.get("/api/health").json()
    assert d["status"] == "ok" and d["chunks"] == 1


def test_ask_stream_emits_sources_then_tokens(client):
    r = client.post("/api/ask/stream", json={"question": "캐시 왜 24시간?"})
    assert r.status_code == 200
    body = r.text
    assert "event: sources" in body
    assert body.count("event: token") == 2      # 토큰 2개 스트리밍
    assert "event: done" in body


def test_ask_returns_503_on_ollama_down(client, rag_mod):
    def boom(_prompt):
        raise rag_mod.OllamaError("연결 실패")
    rag_mod.generate = boom
    r = client.post("/api/ask", json={"question": "캐시"})
    assert r.status_code == 503
    assert "error" in r.json()
