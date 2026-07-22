"""
경량 BM25 — 어휘(lexical) 검색. 외부 의존성 없이 numpy 만 사용.

임베딩(의미 검색)은 'EasyPaymentController', 에러코드, 'filesort' 같은
정확한 식별자를 잘 못 잡는다. BM25 는 그 토큰을 직접 매칭하므로
둘을 합치면(하이브리드) 코드/로그 검색 품질이 올라간다.
"""
from __future__ import annotations

import math
import re

import numpy as np

# 영문/숫자/밑줄 식별자는 통째로, 한글은 음절 덩어리로 토큰화한다.
_TOKEN_RE = re.compile(r"[0-9a-zA-Z_]+|[가-힣]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25. 코퍼스(토큰 리스트들)로 한 번 빌드하고 질의마다 scores() 호출."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.doclen = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doclen.mean()) if self.N else 0.0

        # 역색인(postings): term -> [(doc_index, term_freq)], 그리고 df
        self.postings: dict[str, list[tuple[int, int]]] = {}
        df: dict[str, int] = {}
        for i, toks in enumerate(corpus_tokens):
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                self.postings.setdefault(t, []).append((i, f))
                df[t] = df.get(t, 0) + 1

        # idf (음수 방지를 위해 +1 한 형태)
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """질의 토큰에 대한 문서별 BM25 점수 벡터(len=N)."""
        s = np.zeros(self.N, dtype=np.float32)
        if not self.N or self.avgdl == 0:
            return s
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                denom = f + self.k1 * (1 - self.b + self.b * self.doclen[i] / self.avgdl)
                s[i] += idf * (f * (self.k1 + 1)) / denom
        return s
