#!/usr/bin/env python3
"""
MCP 서버 — Claude Code / Desktop 안에서 과거 대화 로그를 바로 검색한다.

웹(server.py)·CLI(rag.py)와 똑같이 rag.py 의 로직과 같은 인덱스(index.npz)를 쓴다.
입구가 하나 더 생기는 것일 뿐, 셋은 동시에 공존한다.

실행(보통은 MCP 클라이언트가 자동 기동):
  pip install -e ".[mcp]"
  python3 mcp_server.py            # stdio transport

Claude 설정 등록 예시는 README "MCP 로 붙여 쓰기" 참고.
인덱스가 있어야 한다 — 먼저 `rag.py ingest`. 임베딩에 Ollama 가 떠 있어야 한다.

주의: stdio transport 에서는 stdout 이 프로토콜 채널이다. rag 의 print 는 stderr 로 돌린다.
"""
import contextlib
import sys

# mcp 2.x 는 FastMCP 를 MCPServer 로 바꿨다(tool 데코레이터·run 은 같다). 1.x 도 지원한다.
try:
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[attr-defined,no-redef]

import notes
import rag

mcp = _Server("chatlog")

# 인덱스/BM25 는 첫 호출 때 한 번 로드해 캐시한다(웹서버와 같은 패턴).
_STATE: dict = {"mat": None, "metas": None, "bm25": None, "loaded": False}


def _ensure_loaded() -> int | None:
    """인덱스를 메모리에 적재(최초 1회). 반환: 청크 수, 없으면 None."""
    if not _STATE["loaded"]:
        mat, metas = rag.load_index()
        _STATE["mat"], _STATE["metas"] = mat, metas
        _STATE["bm25"] = rag.build_bm25(metas) if metas else None
        _STATE["loaded"] = True
    return None if _STATE["metas"] is None else len(_STATE["metas"])


def _run_search(question, k, alpha, topic, since, until):
    """공통 검색 → (hits, note). hits 가 None 이면 note 에 안내문."""
    if _ensure_loaded() is None:
        return None, "인덱스가 없습니다. 먼저 `python3 rag.py ingest` 를 실행하세요."
    bm25 = _STATE["bm25"] if alpha < 1.0 else None
    hits, max_sim = rag.search(question, _STATE["mat"], _STATE["metas"], k=k,
                               topic=topic, since=since, until=until, alpha=alpha, bm25=bm25)
    if hits is None:
        return None, "필터(topic/기간) 조건에 맞는 기록이 없습니다."
    if not hits:
        return None, f"관련 기록을 찾지 못했습니다 (최고 유사도 {max_sim:.2f})."
    return hits, None


@mcp.tool()
def search_chatlog(question: str, k: int = rag.TOP_K, alpha: float = rag.VEC_WEIGHT,
                   topic: str | None = None, since: str | None = None,
                   until: str | None = None) -> str:
    """과거 AI 대화 로그에서 질문과 관련된 기록을 찾아 (날짜·출처와 함께) 반환한다.

    로컬 LLM 생성 없이 검색된 원문 청크만 돌려주므로 빠르다 — 답변 요약은 호출자(Claude)가 한다.
    "예전에 그거 왜 그렇게 했더라?" 처럼 사용자의 과거 결정·맥락을 물을 때 사용한다.

    Args:
        question: 한국어 질문/검색어
        k: 가져올 기록 수 (기본 6)
        alpha: 하이브리드 가중치 1=의미(벡터)/0=어휘(BM25) (기본 0.5)
        topic: 주제로 사전 필터 (예: caching) — 완전일치
        since: 이 날짜 이후만 (YYYY-MM-DD)
        until: 이 날짜 이전만 (YYYY-MM-DD)
    """
    try:
        hits, note = _run_search(question, k, alpha, topic, since, until)
    except rag.OllamaError as e:
        return f"[오류] {e}"
    if hits is None:
        return note
    blocks = [
        f"{rag.note_mark(m)}[{m['date']} · {m['source']} · {m['topic']} · {m['file']}] "
        f"(유사도 {s:.2f})\n{m['text']}"
        for m, s in hits
    ]
    return f"관련 기록 {len(hits)}건:\n\n" + "\n\n---\n\n".join(blocks)


@mcp.tool()
def ask_chatlog(question: str, k: int = rag.TOP_K, alpha: float = rag.VEC_WEIGHT,
                topic: str | None = None, since: str | None = None,
                until: str | None = None) -> str:
    """과거 대화 로그를 검색해 로컬 LLM(qwen2.5)이 직접 한국어로 답을 생성한다.

    검색 결과를 호출자 대신 로컬 모델이 요약·정리해 주길 원할 때 사용한다.
    (단순히 원문 기록만 필요하면 search_chatlog 가 더 빠르다.) Ollama 가 떠 있어야 한다.

    Args: search_chatlog 와 동일.
    """
    try:
        hits, note = _run_search(question, k, alpha, topic, since, until)
        if hits is None:
            return note
        answer = rag.generate(rag.build_prompt(question, hits))
    except rag.OllamaError as e:
        return f"[오류] {e}"
    sources = "\n".join(f"  · {rag.note_mark(m)}{m['date']} {m['file']} (유사도 {s:.2f})"
                        for m, s in hits)
    return f"{answer}\n\n── 참고한 기록 ──\n{sources}"


@mcp.tool()
def save_note(title: str, text: str, sources: list[str] | None = None,
              topic: str | None = None, tags: list[str] | None = None) -> str:
    """확인한 결론을 출처와 함께 결론 메모 파일(notes/notes.jsonl)에 저장하고 바로 인덱스에 넣는다.

    긴 원문을 다시 뒤지지 않게, 질문에 답이 나온 시점에 "결론 + 출처 링크"를 남길 때 사용한다.
    이후 search_chatlog 에서 이 메모가 원문 기록보다 앞에 온다.
    저장 전에 토큰·키 같은 비밀값은 마스킹된다.

    Args:
        title: 한 줄 제목 (예: "11번가 Open API 키 발급 경로")
        text: 결론 본문 — 다음 사람이 이것만 읽어도 되게 쓴다
        sources: 출처 링크·문서 번호·Jira 키 목록
        topic: 주제 (search_chatlog 의 topic 필터와 같은 축). 기본 note
        tags: 태그 목록
    """
    try:
        rec = notes.add_note(title, text, sources=sources, topic=topic, tags=tags)
    except ValueError as e:
        return f"[오류] {e}"
    try:
        # ingest 의 print 가 stdout(프로토콜 채널)을 더럽히지 않게 stderr 로 돌린다
        with contextlib.redirect_stdout(sys.stderr):
            rag.cmd_ingest(None)
        _STATE["loaded"] = False
        n = _ensure_loaded()
        status = f"인덱스 갱신 완료({n} 청크)"
    except rag.OllamaError as e:
        status = ("저장은 됐지만 인덱스 갱신은 실패했습니다. "
                  f"Ollama 를 띄운 뒤 reload_chatlog 를 호출하세요 ({e})")
    masked = f", 비밀값 {rec['redacted']}건 마스킹" if rec.get("redacted") else ""
    return f"결론 메모 저장: [{rec['date']}] {rec['title']} (id {rec['id']}{masked}). {status}"


@mcp.tool()
def list_notes(topic: str | None = None, since: str | None = None, limit: int = 20) -> str:
    """저장된 결론 메모를 최신순으로 보여준다(검색 없이 목록만). topic·since 로 거른다."""
    items = notes.list_notes(topic=topic, since=since, limit=limit)
    if not items:
        return "저장된 결론 메모가 없습니다."
    blocks = []
    for n in items:
        block = f"[{n['date']} · {n['topic']} · id {n['id']}] {n['title']}\n{n['text']}"
        if n.get("sources"):
            block += "\n출처: " + " · ".join(n["sources"])
        blocks.append(block)
    return f"결론 메모 {len(items)}건:\n\n" + "\n\n---\n\n".join(blocks)


@mcp.tool()
def reload_chatlog() -> str:
    """인덱스를 다시 읽어들인다(로그·메모를 바꾸고 `rag.py ingest` 를 돌린 뒤)."""
    _STATE["loaded"] = False
    n = _ensure_loaded()
    if n is None:
        return "인덱스가 없습니다 (rag.py ingest 필요)."
    return f"인덱스 재적재 완료: {n} 청크"


if __name__ == "__main__":
    mcp.run()
