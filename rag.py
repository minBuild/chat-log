#!/usr/bin/env python3
"""
chatlog : AI 대화 로그를 한국어로 검색하는 최소 RAG 시스템

구조:
  logs/*.jsonl  →  (ingest) 임베딩 인덱스  →  (ask) 한국어 질문 답변

의존성: numpy 하나. 임베딩/생성은 모두 로컬 Ollama가 담당.
  - 임베딩 모델: bge-m3 (다국어, 한국어 우수)
  - 생성 모델 : 환경변수 CHATLOG_LLM (기본 qwen2.5:7b)

사용법:
  python3 rag.py ingest                  # logs/ 를 읽어 인덱스 생성
  python3 rag.py ask "작년에 그 캐시 TTL 왜 24시간으로 잡았더라?"
  python3 rag.py ask "..." --topic caching --k 8
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import sys
import urllib.request
from collections.abc import Iterator

import numpy as np

from bm25 import BM25, tokenize

# ── 설정 ───────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
INDEX_NPZ = os.path.join(HERE, "index.npz")
INDEX_META = os.path.join(HERE, "index_meta.json")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("CHATLOG_EMBED", "bge-m3")
LLM_MODEL = os.environ.get("CHATLOG_LLM", "qwen2.5:7b")

CHUNK_SIZE = 800        # 청크 길이(문자)
CHUNK_OVERLAP = 150     # 청크 간 겹침(문자) — 문맥 단절 방지
TOP_K = 6               # 검색 시 가져올 청크 수
MIN_SIM = 0.45          # 이 (벡터) 유사도 미만 청크는 버림 — 무관한 기록의 오염 방지
VEC_WEIGHT = 0.5        # 하이브리드 가중치 α: 1=순수 벡터, 0=순수 BM25, 0.5=반반


# ── Ollama 호출 (stdlib 만 사용) ────────────────────────────────────
class OllamaError(RuntimeError):
    """Ollama 서버에 닿지 못하거나 응답이 비정상일 때. (CLI는 메시지 출력 후 종료, 서버는 503)"""


def _request(path: str, payload: dict) -> urllib.request.Request:
    return urllib.request.Request(
        OLLAMA_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def _post(path: str, payload: dict) -> dict:
    try:
        with urllib.request.urlopen(_request(path, payload), timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise OllamaError(f"Ollama 연결 실패 ({OLLAMA_URL}). 'ollama serve' 가 떠 있나요? — {e}")


def embed(text: str) -> np.ndarray:
    out = _post("/api/embeddings", {"model": EMBED_MODEL, "prompt": text})
    v = np.array(out["embedding"], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v          # 정규화 → 내적이 곧 코사인 유사도


def generate(prompt: str) -> str:
    out = _post("/api/generate", {"model": LLM_MODEL, "prompt": prompt, "stream": False})
    return out["response"].strip()


def generate_stream(prompt: str) -> Iterator[str]:
    """답변 토큰을 스트리밍으로 yield (웹 UI 실시간 타이핑용)."""
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": True}
    try:
        with urllib.request.urlopen(_request("/api/generate", payload), timeout=300) as resp:
            for raw in resp:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                if obj.get("response"):
                    yield obj["response"]
                if obj.get("done"):
                    break
    except urllib.error.URLError as e:
        raise OllamaError(f"Ollama 연결 실패 ({OLLAMA_URL}) — {e}")


# ── 청킹 ───────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """문자 단위 분할 — 단일 메시지가 한도를 넘을 때만 쓰는 폴백."""
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + CHUNK_SIZE])
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def chunk_records(records: list[dict]) -> list[str]:
    """턴(메시지) 경계를 지키는 청킹. 문장 중간을 자르지 않아 Q&A 쌍이 함께 묶인다.
    한 메시지가 한도를 넘으면 그 메시지만 문자 분할하고, 청크 간 1메시지를 겹쳐 문맥을 잇는다."""
    lines = [f"{r.get('role', '?')}: {r.get('text', '')}" for r in records]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for line in lines:
        if len(line) > CHUNK_SIZE:               # 초대형 메시지 → 문자 분할 폴백
            if cur:
                chunks.append("\n".join(cur))
                cur, cur_len = [], 0
            chunks.extend(chunk_text(line))
            continue
        if cur and cur_len + len(line) + 1 > CHUNK_SIZE:
            chunks.append("\n".join(cur))
            cur = cur[-1:]                       # 직전 메시지 1개 겹침
            cur_len = len(cur[0]) + 1
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def iter_log_chunks(log_dir: str | None = None) -> Iterator[tuple[str, dict]]:
    """logs/*.jsonl → (청크 텍스트, 메타) 스트림. ingest 와 eval 이 공유한다."""
    for path in sorted(glob.glob(os.path.join(log_dir or LOG_DIR, "*.jsonl"))):
        recs = load_records(path)
        if not recs:
            continue
        head = recs[0]
        for ci, chunk in enumerate(chunk_records(recs)):
            yield chunk, {"file": os.path.basename(path), "date": head.get("date", ""),
                          "topic": head.get("topic", ""), "source": head.get("source", ""),
                          "chunk": ci, "text": chunk}


def load_records(path: str) -> list[dict]:
    """jsonl 파일 한 개를 읽어 레코드 리스트로 반환."""
    recs = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  ! {os.path.basename(path)}:{ln} JSON 파싱 실패, 건너뜀")
    return recs


# ── ingest: 인덱스 생성 ────────────────────────────────────────────
def _load_cache() -> dict:
    """기존 인덱스 → {청크 해시: 벡터} 캐시. 변경 없는 청크 재사용용."""
    if not (os.path.exists(INDEX_NPZ) and os.path.exists(INDEX_META)):
        return {}
    old = np.load(INDEX_NPZ)["embeddings"]
    with open(INDEX_META, encoding="utf-8") as f:
        metas = json.load(f)
    return {_hash(m["text"]): old[i] for i, m in enumerate(metas)}


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cmd_ingest(_args) -> None:
    pairs = list(iter_log_chunks())
    if not pairs:
        sys.exit(f"[오류] {LOG_DIR} 에 *.jsonl 로그가 없습니다.")

    cache = _load_cache()
    vectors, metas = [], []
    reused = embedded = 0
    for chunk, meta in pairs:
        h = _hash(chunk)
        if h in cache:                           # 내용 동일 → 임베딩 재사용
            vectors.append(cache[h])
            reused += 1
        else:                                    # 새로/바뀐 청크만 임베딩
            vectors.append(embed(chunk))
            embedded += 1
        metas.append(meta)

    nfiles = len({m["file"] for m in metas})
    print(f"  {nfiles} 파일 → {len(metas)} 청크")
    mat = np.vstack(vectors).astype(np.float32)
    np.savez(INDEX_NPZ, embeddings=mat)
    with open(INDEX_META, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)
    print(f"\n인덱스 생성 완료: {len(metas)} 청크 "
          f"(신규 임베딩 {embedded}, 재사용 {reused}), 차원 {mat.shape[1]}")


# ── 검색 (출력과 분리: 테스트·평가 하네스가 재사용) ────────────────
def load_index() -> tuple[np.ndarray | None, list | None]:
    """인덱스를 메모리에 로드. 없으면 (None, None)."""
    if not (os.path.exists(INDEX_NPZ) and os.path.exists(INDEX_META)):
        return None, None
    mat = np.load(INDEX_NPZ)["embeddings"]
    with open(INDEX_META, encoding="utf-8") as f:
        metas = json.load(f)
    return mat, metas


def build_bm25(metas: list[dict]) -> BM25:
    """청크 텍스트로 BM25 어휘 인덱스 빌드."""
    return BM25([tokenize(m["text"]) for m in metas])


def _minmax(x: np.ndarray) -> np.ndarray:
    """질의 단위 min-max 정규화 → 벡터/BM25 점수를 같은 [0,1] 축으로."""
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(x)


def search(question: str, mat: np.ndarray, metas: list[dict], k: int = TOP_K,
           min_sim: float = MIN_SIM, topic: str | None = None, since: str | None = None,
           until: str | None = None, alpha: float = VEC_WEIGHT,
           bm25: BM25 | None = None) -> tuple[list | None, float | None]:
    """질문 → 하이브리드(벡터+BM25) 검색. 반환: (hits, max_sim).

    랭킹은 α·벡터 + (1-α)·BM25 결합 점수로, 정렬·노출되는 점수(meta 옆)는 벡터 코사인.
    필터: 벡터 코사인이 min_sim 이상이거나, 어휘 매칭(BM25>0)이 있으면 유지
    → 임베딩이 못 잡는 정확한 식별자 매칭도 살아남는다.
    필터로 후보가 0이면 (None, None)."""
    idx = list(range(len(metas)))
    if topic:
        idx = [i for i in idx if metas[i]["topic"] == topic]
    if since:
        idx = [i for i in idx if metas[i]["date"] >= since]
    if until:
        idx = [i for i in idx if metas[i]["date"] <= until]
    if not idx:
        return None, None

    qv = embed(question)
    vec = mat[idx] @ qv                          # 정규화된 벡터 → 내적이 곧 코사인
    lex = None
    if bm25 is not None and alpha < 1.0:
        lex = bm25.scores(tokenize(question))[idx]
        combined = alpha * _minmax(vec) + (1 - alpha) * _minmax(lex)
    else:
        combined = vec

    order = np.argsort(-combined)[:k]
    hits = []
    for j in order:
        if vec[j] >= min_sim or (lex is not None and lex[j] > 0):
            hits.append((metas[idx[j]], float(vec[j])))
    return hits, float(vec.max())


# ── 프롬프트 (CLI·API 공용) ────────────────────────────────────────
def build_prompt(question: str, hits: list) -> str:
    context = "\n\n".join(
        f"[{m['date']} · {m['source']} · {m['topic']}] (유사도 {s:.2f})\n{m['text']}"
        for m, s in hits
    )
    today = datetime.date.today().isoformat()
    return f"""너는 사용자의 과거 AI 대화 로그를 검색해 답하는 한국어 비서다.
오늘 날짜는 {today} 이다. '작년','지난달' 같은 표현은 이 날짜를 기준으로 해석해라.
아래 '검색된 기록'만 근거로 사용해서 질문에 답해라.

규칙:
- 반드시 한국어로만 답해라. 다른 언어를 절대 섞지 마라.
- 기록에 근거가 있으면 날짜를 함께 인용해라.
- 기록에서 답을 찾을 수 없을 때만 "기록에서 찾지 못했다"고 말해라. 지어내지 마라.

[검색된 기록]
{context}

[질문]
{question}

[답변]"""


# ── ask: 질문 답변 ─────────────────────────────────────────────────
def cmd_ask(args) -> None:
    mat, metas = load_index()
    if mat is None or metas is None:
        sys.exit("[오류] 인덱스가 없습니다. 먼저 'python3 rag.py ingest' 를 실행하세요.")

    bm25 = build_bm25(metas) if args.alpha < 1.0 else None
    hits, max_sim = search(args.question, mat, metas, k=args.k, min_sim=args.min_sim,
                           topic=args.topic, since=args.since, until=args.until,
                           alpha=args.alpha, bm25=bm25)
    if hits is None:
        sys.exit("[안내] 필터 조건에 맞는 로그가 없습니다.")
    if not hits:
        print(f"\n(모델: {LLM_MODEL})\n기록에서 관련 내용을 찾지 못했습니다. "
              f"(최고 유사도 {max_sim:.2f} < 임계값 {args.min_sim})")
        return

    prompt = build_prompt(args.question, hits)
    print(f"\n(모델: {LLM_MODEL}, 검색 청크: {len(hits)}개)\n")
    print(generate(prompt))
    print("\n── 참고한 기록 ──")
    for m, s in hits:
        print(f"  · {m['date']} {m['file']} (유사도 {s:.2f})")


# ── 엔트리포인트 ───────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="AI 대화 로그 한국어 RAG 검색")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="logs/ 를 읽어 인덱스 생성")
    a = sub.add_parser("ask", help="한국어로 질문")
    a.add_argument("question")
    a.add_argument("--k", type=int, default=TOP_K, help="검색 청크 수")
    a.add_argument("--min-sim", type=float, default=MIN_SIM, dest="min_sim",
                   help=f"이 유사도 미만 청크는 버림 (기본 {MIN_SIM})")
    a.add_argument("--alpha", type=float, default=VEC_WEIGHT,
                   help=f"하이브리드 가중치 1=벡터/0=BM25 (기본 {VEC_WEIGHT})")
    a.add_argument("--topic", help="주제로 사전 필터 (예: caching)")
    a.add_argument("--since", help="이 날짜 이후 (YYYY-MM-DD)")
    a.add_argument("--until", help="이 날짜 이전 (YYYY-MM-DD)")
    args = p.parse_args()
    try:
        {"ingest": cmd_ingest, "ask": cmd_ask}[args.cmd](args)
    except OllamaError as e:
        sys.exit(f"[오류] {e}")


if __name__ == "__main__":
    main()
