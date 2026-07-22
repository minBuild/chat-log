#!/usr/bin/env python3
"""
평가 하네스 — 검색(retrieval) 품질을 recall@k / MRR 로 측정한다.

라벨된 질문 세트(eval/qa.jsonl: {"q":..., "expect_topic":...})에 대해
검색 상위 k개 안에 기대 topic 의 청크가 들어오는지를 본다.
임계값·k·임베딩 모델, 그리고 하이브리드 가중치 α 의 효과를 숫자로 비교한다.

사용법 (Ollama 가 떠 있어야 함 — 실제 임베딩 사용):
  python3 eval.py                  # 기본 α 로 1회
  python3 eval.py --sweep          # α=1(벡터)/0.5(하이브리드)/0(BM25) 비교표
  python3 eval.py --logs logs --alpha 0.3 --k 10
"""
import argparse
import os

import numpy as np

import rag


def build_index(logs_dir):
    """logs_dir 의 jsonl 들을 메모리 인덱스로 빌드(디스크 인덱스와 독립, ingest 와 동일 청킹)."""
    vectors, metas = [], []
    for chunk, meta in rag.iter_log_chunks(logs_dir):
        vectors.append(rag.embed(chunk))
        metas.append(meta)
    if not vectors:
        raise SystemExit(f"[오류] {logs_dir} 에 로그가 없습니다.")
    return np.vstack(vectors).astype("float32"), metas


def first_relevant_rank(hits, expect_topic):
    """상위 결과에서 기대 topic 이 처음 등장하는 순위(1-base). 없으면 None."""
    for rank, (m, _score) in enumerate(hits, 1):
        if m["topic"] == expect_topic:
            return rank
    return None


def evaluate(mat, metas, qas, k, alpha, bm25):
    """(recall dict, mrr, ranks) 반환."""
    ks = [x for x in (1, 3, 5, 10) if x <= k]
    hit = {x: 0 for x in ks}
    rr_sum, ranks = 0.0, []
    for qa in qas:
        hits, _ = rag.search(qa["q"], mat, metas, k=k, min_sim=-1.0, alpha=alpha, bm25=bm25)
        rank = first_relevant_rank(hits, qa["expect_topic"])
        ranks.append(rank)
        for x in ks:
            if rank and rank <= x:
                hit[x] += 1
        rr_sum += (1.0 / rank) if rank else 0.0
    return {x: hit[x] / len(qas) for x in ks}, rr_sum / len(qas), ranks


def _label(alpha):
    return {1.0: "벡터", 0.0: "BM25"}.get(alpha, f"하이브리드 α={alpha}")


def main():
    ap = argparse.ArgumentParser(description="검색 품질 평가 (recall@k, MRR)")
    ap.add_argument("--logs", default=os.path.join(rag.HERE, "examples"))
    ap.add_argument("--qa", default=os.path.join(rag.HERE, "eval", "qa.jsonl"))
    ap.add_argument("--k", type=int, default=5, help="검색 깊이")
    ap.add_argument("--alpha", type=float, default=rag.VEC_WEIGHT,
                    help="하이브리드 가중치 1=벡터/0=BM25 (기본 0.5)")
    ap.add_argument("--sweep", action="store_true",
                    help="α=1.0/0.5/0.0 을 한 번에 비교")
    args = ap.parse_args()

    mat, metas = build_index(args.logs)
    qas = rag.load_records(args.qa)
    bm25 = rag.build_bm25(metas)
    ks = [x for x in (1, 3, 5, 10) if x <= args.k]

    print(f"\n질문 {len(qas)}개 · 청크 {len(metas)}개 · 임베딩 {rag.EMBED_MODEL}\n")

    if args.sweep:
        cols = "  ".join(f"recall@{x}" for x in ks) + "    MRR"
        print(f"{'모드':<18}{cols}")
        print("─" * (18 + len(cols) + 2))
        for alpha in (1.0, 0.5, 0.0):
            rec, mrr, _ = evaluate(mat, metas, qas, args.k, alpha, bm25)
            cells = "  ".join(f"{rec[x]:>8.2f}" for x in ks)
            print(f"{_label(alpha):<18}{cells}  {mrr:>6.3f}")
        return

    rec, mrr, ranks = evaluate(mat, metas, qas, args.k, args.alpha, bm25)
    print(f"[{_label(args.alpha)}]")
    for qa, rank in zip(qas, ranks):
        q = qa["q"] if len(qa["q"]) <= 34 else qa["q"][:33] + "…"
        print(f"  {q:<36}{qa['expect_topic']:<14}{rank or '✗'}")
    print("─" * 60)
    for x in ks:
        print(f"recall@{x:<2}: {rec[x]:.2f}")
    print(f"MRR     : {mrr:.3f}")


if __name__ == "__main__":
    main()
