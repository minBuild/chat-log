"""
FastAPI 웹 서버 — 브라우저에서 한국어로 묻고 답을 받는다.

실행:
  pip install -e ".[web]"      # fastapi, uvicorn
  python3 server.py            # http://127.0.0.1:8000
  (인덱스가 있어야 한다 — 먼저 rag.py ingest)

인덱스와 BM25 는 시작 시 한 번만 로드한다. 로그를 갱신했으면 POST /api/reload.
"""
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import rag

STATE = {"mat": None, "metas": None, "bm25": None}


def _load():
    mat, metas = rag.load_index()
    STATE["mat"], STATE["metas"] = mat, metas
    STATE["bm25"] = rag.build_bm25(metas) if metas else None
    return 0 if metas is None else len(metas)


@asynccontextmanager
async def lifespan(_app):
    _load()
    yield


app = FastAPI(title="chatlog", lifespan=lifespan)


class AskReq(BaseModel):
    question: str
    k: int = rag.TOP_K
    alpha: float = rag.VEC_WEIGHT
    topic: str | None = None
    generate: bool = True          # False 면 검색 결과만(빠름), LLM 생성 생략


def _retrieve(req):
    """검색만 수행 → (hits, sources, note). 인덱스 없으면 hits=None."""
    if STATE["mat"] is None:
        return None, [], "인덱스가 없습니다. 먼저 'python3 rag.py ingest' 를 실행하세요."
    bm25 = STATE["bm25"] if req.alpha < 1.0 else None
    hits, max_sim = rag.search(req.question, STATE["mat"], STATE["metas"],
                               k=req.k, alpha=req.alpha, topic=req.topic, bm25=bm25)
    if hits is None:
        return [], [], "필터 조건에 맞는 기록이 없습니다."
    if not hits:
        return [], [], f"관련 기록을 찾지 못했습니다 (최고 유사도 {max_sim:.2f})."
    sources = [{"date": m["date"], "file": m["file"], "topic": m["topic"],
                "score": round(s, 3), "text": m["text"][:240]} for m, s in hits]
    return hits, sources, None


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(rag.HERE, "web", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    n = STATE["metas"] and len(STATE["metas"])
    return {"status": "ok" if STATE["mat"] is not None else "no-index",
            "chunks": n or 0, "model": rag.LLM_MODEL, "embed": rag.EMBED_MODEL}


@app.post("/api/ask")
def ask(req: AskReq):
    hits, sources, note = _retrieve(req)
    if hits is None:
        return JSONResponse({"error": note}, status_code=400)
    if not hits:
        return {"answer": None, "sources": [], "note": note}
    try:
        answer = rag.generate(rag.build_prompt(req.question, hits)) if req.generate else None
    except rag.OllamaError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    return {"answer": answer, "sources": sources, "note": None}


def _sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/ask/stream")
def ask_stream(req: AskReq):
    """SSE 스트림: sources 이벤트 1개 → token 이벤트 다수 → done. 웹 UI 실시간 타이핑용."""
    hits, sources, note = _retrieve(req)
    if hits is None:
        return JSONResponse({"error": note}, status_code=400)

    def gen():
        if not hits:
            yield _sse("note", note)
            yield _sse("done", "")
            return
        yield _sse("sources", sources)
        try:
            for tok in rag.generate_stream(rag.build_prompt(req.question, hits)):
                yield _sse("token", tok)
        except rag.OllamaError as e:
            yield _sse("error", str(e))
        yield _sse("done", "")

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/reload")
def reload():
    return {"chunks": _load()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,
                host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))
