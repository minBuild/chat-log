#!/usr/bin/env python3
"""
import_logs : 외부 대화 데이터를 chatlog 의 logs/*.jsonl 로 변환

지원 소스:
  claude-code     Claude Code 세션 transcript (~/.claude/projects/**/*.jsonl)
                  파일 1개 / 프로젝트 폴더 / 전체 projects 디렉토리 모두 가능
  claude-export   claude.ai '데이터 내보내기' 의 conversations.json
                  (설정 → 계정 → 데이터 내보내기 → 메일로 받은 zip 안)
  memory          클로드가 저장한 메모리 폴더 (*.md, frontmatter 포함)

사용법:
  python3 import_logs.py claude-code ~/.claude/projects                 # 전체
  python3 import_logs.py claude-code ~/.claude/projects/-Users-...-myproject
  python3 import_logs.py claude-export ~/Downloads/conversations.json
  python3 import_logs.py memory ~/.claude/projects/<proj>/memory
  python3 import_logs.py claude-export conv.json --topic caching   # 주제 강제 지정

변환 후 'python3 rag.py ingest' 로 인덱스를 갱신하면 된다.
"""
import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")


def slugify(s, maxlen=30):
    """제목 → 파일명에 안전한 슬러그 (한글 보존)."""
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"[^0-9A-Za-z가-힣\-_]", "", s)
    s = re.sub(r"-+", "-", s)                  # 연속 하이픈 합치기
    return (s[:maxlen] or "untitled").strip("-")


def unique_path(date, slug):
    """date_slug.jsonl 경로, 충돌 시 -2, -3 ... 붙임."""
    base = f"{date}_{slug}"
    path = os.path.join(LOG_DIR, base + ".jsonl")
    n = 2
    while os.path.exists(path):
        path = os.path.join(LOG_DIR, f"{base}-{n}.jsonl")
        n += 1
    return path


def msg_text(m):
    """메시지에서 본문 추출 — text 우선, 없으면 content 블록 합침."""
    if m.get("text"):
        return m["text"]
    parts = [b.get("text", "") for b in m.get("content", []) if b.get("type") == "text"]
    return "\n".join(p for p in parts if p)


# ── claude.ai export ───────────────────────────────────────────────
def import_claude_export(path, topic_override):
    with open(path, encoding="utf-8") as f:
        convs = json.load(f)
    if isinstance(convs, dict):           # 단일 대화로 내보낸 경우 대비
        convs = [convs]

    written = 0
    for c in convs:
        msgs = c.get("chat_messages", [])
        if not msgs:
            continue
        date = (c.get("created_at") or "")[:10] or "0000-00-00"
        title = c.get("name") or "untitled"
        topic = topic_override or slugify(title, 20)
        out = unique_path(date, slugify(title))
        with open(out, "w", encoding="utf-8") as f:
            for m in msgs:
                text = msg_text(m).strip()
                if not text:
                    continue
                role = "user" if m.get("sender") == "human" else "assistant"
                f.write(json.dumps({
                    "date": (m.get("created_at") or date)[:10],
                    "source": "claude",
                    "topic": topic,
                    "role": role,
                    "text": text,
                }, ensure_ascii=False) + "\n")
        written += 1
        print(f"  ✓ {os.path.basename(out)}  ({len(msgs)} 메시지)")
    print(f"\n{written} 개 대화 → logs/ 변환 완료")


# ── 클로드 메모리 폴더 ─────────────────────────────────────────────
def parse_frontmatter(raw):
    """간단 frontmatter 파서 → (meta dict, body)."""
    if not raw.startswith("---"):
        return {}, raw
    _, fm, body = raw.split("---", 2)
    meta = {}
    for line in fm.splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body.strip()


def import_memory(path, topic_override):
    files = sorted(glob.glob(os.path.join(path, "*.md")))
    files = [f for f in files if os.path.basename(f) != "MEMORY.md"]
    if not files:
        print(f"[안내] {path} 에 *.md 메모리가 없습니다.")
        return
    written = 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            meta, body = parse_frontmatter(f.read())
        name = meta.get("name") or os.path.splitext(os.path.basename(fp))[0]
        date = __import__("datetime").date.fromtimestamp(os.path.getmtime(fp)).isoformat()
        desc = meta.get("description", "")
        out = unique_path(date, slugify("memory-" + name))
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "date": date,
                "source": "claude-memory",
                "topic": topic_override or meta.get("type", "memory"),
                "role": "memory",
                "text": (desc + "\n" + body).strip() if desc else body,
            }, ensure_ascii=False) + "\n")
        written += 1
        print(f"  ✓ {os.path.basename(out)}")
    print(f"\n{written} 개 메모리 → logs/ 변환 완료")


# ── Claude Code 세션 transcript ────────────────────────────────────
def _cc_assistant_text(content):
    """assistant content 블록에서 text 만 추출 (thinking/tool_use 제외)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
    return ""


def _cc_parse_session(path):
    """세션 jsonl → (title, topic, [records]). 도구 노이즈는 버린다."""
    title, cwd, recs = None, None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = o.get("type")
        if t == "ai-title":
            title = o.get("aiTitle") or title
            continue
        if t not in ("user", "assistant"):
            continue
        m = o.get("message")
        if not isinstance(m, dict):
            continue
        cwd = cwd or o.get("cwd")
        ts = (o.get("timestamp") or "")[:10]
        content = m.get("content")
        if t == "user":
            if not isinstance(content, str):     # list = tool_result → 노이즈
                continue
            text = content.strip()
        else:
            text = _cc_assistant_text(content).strip()
        # 하네스가 끼워넣는 명령/캐럿 줄은 스킵
        if not text or text.startswith(("<command-", "Caveat:")):
            continue
        recs.append({"date": ts, "role": t, "text": text})
    topic = os.path.basename(cwd) if cwd else "claude-code"
    return title, topic, recs


def import_claude_code(path, topic_override, prune=False):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
    else:
        files = [path]
    if not files:
        print(f"[안내] {path} 에서 *.jsonl 세션을 찾지 못했습니다.")
        return
    written, skipped = set(), 0
    for fp in files:
        title, topic, recs = _cc_parse_session(fp)
        if len(recs) < 2:                        # 실질 대화 없는 세션 스킵
            skipped += 1
            continue
        date = recs[0]["date"] or "0000-00-00"
        title = title or recs[0]["text"][:20]
        # 세션 id(전체 uuid) 기반 고정 파일명 → 매일 재실행 시 같은 세션을 덮어씀(멱등)
        sessid = os.path.splitext(os.path.basename(fp))[0]
        name = f"{date}_{slugify(title)}__{sessid}.jsonl"
        with open(os.path.join(LOG_DIR, name), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps({
                    "date": r["date"] or date,
                    "source": "claude-code",
                    "topic": topic_override or topic,
                    "role": r["role"],
                    "text": r["text"],
                }, ensure_ascii=False) + "\n")
        written.add(name)
    print(f"\n{len(written)} 개 세션 변환, {skipped} 개 스킵(대화 없음) → logs/")

    if prune:                                    # 원본에서 사라진 세션은 인덱스에서도 제거
        pruned = _prune_claude_code(written)
        if pruned:
            print(f"  prune: 사라진 세션 {pruned}개 삭제")


# claude-code 가 만든 파일명 패턴: <날짜>_<슬러그>__<세션uuid>.jsonl
_CC_FILE_RE = re.compile(r"__[0-9a-fA-F-]{6,}\.jsonl$")


def _prune_claude_code(keep):
    """claude-code 산출물 중 이번 실행에서 만들지 않은 것(원본 세션 삭제됨)을 지운다.
    claude-export/memory 산출물은 패턴이 달라 건드리지 않는다."""
    removed = 0
    for fn in os.listdir(LOG_DIR):
        if _CC_FILE_RE.search(fn) and fn not in keep:
            os.remove(os.path.join(LOG_DIR, fn))
            removed += 1
    return removed


def main():
    p = argparse.ArgumentParser(description="외부 대화 데이터 → logs/*.jsonl 변환")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in [("claude-code", "Claude Code 세션 transcript"),
                        ("claude-export", "claude.ai conversations.json"),
                        ("memory", "클로드 메모리 폴더")]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("path")
        sp.add_argument("--topic", help="모든 레코드의 topic 을 이 값으로 강제")
        if name == "claude-code":
            sp.add_argument("--prune", action="store_true",
                            help="원본에서 사라진 세션의 로그도 삭제(삭제 동기화)")
    args = p.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    if args.cmd == "claude-code":
        import_claude_code(args.path, args.topic, prune=args.prune)
    elif args.cmd == "claude-export":
        import_claude_export(args.path, args.topic)
    else:
        import_memory(args.path, args.topic)


if __name__ == "__main__":
    main()
