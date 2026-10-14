#!/usr/bin/env python3
"""
notes : 결론 메모 — 답과 출처를 파일 하나에 남겨 다음엔 원문을 뒤지지 않고 바로 찾는다.

세션 원문은 수백만 자에 같은 페이지가 반복되고 JSON 이스케이프까지 섞여 읽기 어렵다.
그래서 "결론 + 출처 링크"만 `notes/notes.jsonl` 에 한 줄씩 쌓는다(append-only 백업 파일).
ingest 때 메모 하나가 청크 하나로 인덱스에 들어가고, 검색에서는 임계값을 넘긴 메모가
원문 기록보다 앞에 온다. Notion·Confluence 같은 외부 저장소 대신 이 저장소 안의 파일 하나가
기준이다(gitignore — 커밋되지 않음).

사용법:
  python3 notes.py add "11번가 키 발급" "셀러오피스 로그인 뒤 API 관리에서 받는다" \
      --source https://openapi.11st.co.kr --topic lowpick --tag api
  python3 notes.py list [--topic lowpick] [--since 2026-10-01] [--limit 20]
  python3 notes.py export            # notes/notes.md 로 사람이 읽는 목록 내보내기
  python3 rag.py ingest              # 메모를 인덱스에 반영

MCP 에서는 save_note / list_notes 도구가 같은 파일을 쓴다(저장 즉시 인덱스 갱신).
저장 전에 비밀값(토큰·키)은 redact.py 가 마스킹한다.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from collections.abc import Iterable, Iterator

from redact import redact_text

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(HERE, "notes")
NOTES_FILE = os.path.join(NOTES_DIR, "notes.jsonl")
EXPORT_MD = os.path.join(NOTES_DIR, "notes.md")
SOURCE = "note"          # 인덱스 메타의 source 값 — 검색 결과에서 메모를 구분·우대한다


def _clean_list(items: Iterable[str] | None) -> list[str]:
    return [s.strip() for s in (items or []) if s and s.strip()]


def add_note(title: str, text: str, sources: Iterable[str] | None = None, topic: str | None = None,
             tags: Iterable[str] | None = None, date: str | None = None,
             notes_file: str | None = None) -> dict:
    """결론 메모 한 건을 파일 끝에 붙인다. 반환: 저장된 레코드(비밀값은 마스킹된 상태)."""
    title, text = title.strip(), text.strip()
    if not title or not text:
        raise ValueError("제목과 결론 본문은 비울 수 없습니다.")
    now = datetime.datetime.now()
    title, n_title = redact_text(title)
    text, n_text = redact_text(text)
    cleaned_sources = [redact_text(s)[0] for s in _clean_list(sources)]
    rec = {
        "id": hashlib.sha1(f"{now.isoformat()}|{title}|{text}".encode("utf-8")).hexdigest()[:12],
        "date": date or now.date().isoformat(),
        "title": title,
        "topic": (topic or "note").strip() or "note",
        "text": text,
        "sources": cleaned_sources,
        "tags": _clean_list(tags),
        "created_at": now.isoformat(timespec="seconds"),
        "redacted": n_title + n_text,
    }
    path = notes_file or NOTES_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_notes(notes_file: str | None = None) -> list[dict]:
    """파일의 모든 메모(저장 순). 깨진 줄은 건너뛴다."""
    path = notes_file or NOTES_FILE
    if not os.path.exists(path):
        return []
    notes: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                notes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return notes


def list_notes(topic: str | None = None, since: str | None = None, until: str | None = None,
               limit: int = 20, notes_file: str | None = None) -> list[dict]:
    """필터에 맞는 메모를 최신순으로 limit 개."""
    notes = load_notes(notes_file)
    if topic:
        notes = [n for n in notes if n.get("topic") == topic]
    if since:
        notes = [n for n in notes if n.get("date", "") >= since]
    if until:
        notes = [n for n in notes if n.get("date", "") <= until]
    return list(reversed(notes))[:limit]


def note_text(n: dict) -> str:
    """인덱스에 넣는 본문. 제목·결론·출처·태그를 한 덩어리로 — 메모 하나가 청크 하나다."""
    lines = [f"결론 메모: {n.get('title', '')}", n.get("text", "")]
    if n.get("sources"):
        lines.append("출처: " + " · ".join(n["sources"]))
    if n.get("tags"):
        lines.append("태그: " + ", ".join(n["tags"]))
    return "\n".join(lines)


def iter_note_chunks(notes_file: str | None = None) -> Iterator[tuple[str, dict]]:
    """notes.jsonl → (청크 텍스트, 메타) 스트림. rag.iter_all_chunks 가 로그 청크 뒤에 붙인다."""
    path = notes_file or NOTES_FILE
    for i, n in enumerate(load_notes(path)):
        chunk = note_text(n)
        yield chunk, {"file": os.path.basename(path), "date": n.get("date", ""),
                      "topic": n.get("topic", ""), "source": SOURCE, "chunk": i, "text": chunk,
                      "title": n.get("title", ""), "note_id": n.get("id", "")}


def to_markdown(notes: Iterable[dict]) -> str:
    """사람이 읽는 목록(최신순). 백업·공유용."""
    out = ["# 결론 메모", ""]
    for n in reversed(list(notes)):
        out.append(f"## {n.get('date', '')} · {n.get('title', '')}  `{n.get('topic', '')}`")
        out.append("")
        out.append(n.get("text", ""))
        for s in n.get("sources", []):
            out.append(f"- 출처: {s}")
        if n.get("tags"):
            out.append("- 태그: " + ", ".join(n["tags"]))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="결론 메모 — 찾은 답과 출처를 한 파일에 남긴다")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="메모 추가")
    a.add_argument("title")
    a.add_argument("text", help="결론 본문")
    a.add_argument("--source", action="append", default=[],
                   help="출처 링크·문서 번호(여러 번 가능)")
    a.add_argument("--topic", help="주제(검색 --topic 필터와 같은 축). 기본 note")
    a.add_argument("--tag", action="append", default=[], help="태그(여러 번 가능)")
    a.add_argument("--date", help="메모 날짜 YYYY-MM-DD (기본 오늘)")
    ls = sub.add_parser("list", help="메모 목록(최신순)")
    ls.add_argument("--topic")
    ls.add_argument("--since")
    ls.add_argument("--until")
    ls.add_argument("--limit", type=int, default=20)
    ex = sub.add_parser("export", help="notes/notes.md 로 내보내기")
    ex.add_argument("--out", default=EXPORT_MD)
    args = p.parse_args()

    if args.cmd == "add":
        rec = add_note(args.title, args.text, sources=args.source, topic=args.topic,
                       tags=args.tag, date=args.date)
        masked = f" (비밀값 {rec['redacted']}건 마스킹)" if rec["redacted"] else ""
        print(f"  ✓ [{rec['date']}] {rec['title']} → {os.path.relpath(NOTES_FILE, HERE)}{masked}")
        print("  인덱스 반영:  python3 rag.py ingest")
    elif args.cmd == "list":
        items = list_notes(args.topic, args.since, args.until, args.limit)
        if not items:
            print("저장된 결론 메모가 없습니다.")
        for n in items:
            print(f"[{n['date']} · {n['topic']} · {n['id']}] {n['title']}\n  {n['text']}")
            for s in n.get("sources", []):
                print(f"  · {s}")
    else:
        notes = load_notes()
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(to_markdown(notes))
        print(f"  ✓ {len(notes)}건 → {args.out}")


if __name__ == "__main__":
    main()
