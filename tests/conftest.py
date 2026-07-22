"""테스트 공용 픽스처.

핵심: Ollama 없이 돌도록 embed/generate 를 가짜로 갈아끼운다.
가짜 임베딩은 단어 가방(bag-of-words) — 같은 단어를 공유하는 텍스트끼리
코사인 유사도가 높게 나오므로, 실제 의미 검색의 동작을 결정적으로 재현한다.
"""
import json
import os
import re

import numpy as np
import pytest

DIM = 4096


def _make_fake_embed():
    vocab = {}

    def embed(text):
        toks = re.findall(r"[0-9a-zA-Z가-힣]+", text.lower())
        v = np.zeros(DIM, dtype=np.float32)
        for w in toks:
            vocab.setdefault(w, len(vocab))
            v[vocab[w] % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    return embed


@pytest.fixture
def rag_env(tmp_path, monkeypatch):
    """rag 모듈을 임시 디렉토리 + 가짜 임베딩으로 격리."""
    import rag
    monkeypatch.setattr(rag, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(rag, "INDEX_NPZ", str(tmp_path / "index.npz"))
    monkeypatch.setattr(rag, "INDEX_META", str(tmp_path / "index_meta.json"))
    os.makedirs(rag.LOG_DIR, exist_ok=True)
    monkeypatch.setattr(rag, "embed", _make_fake_embed())
    monkeypatch.setattr(rag, "generate", lambda prompt: "가짜-답변")
    return rag


def write_log(rag, name, records):
    """logs/ 에 jsonl 로그 한 개를 쓴다."""
    with open(os.path.join(rag.LOG_DIR, name), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
