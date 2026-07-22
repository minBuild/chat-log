# chatlog

![CI](https://github.com/minBuild/chat-log/actions/workflows/ci.yml/badge.svg)

> 과거의 나에게 묻는다 — 매일 쌓인 내 AI 대화를 로컬 LLM이 대신 뒤져 답한다.

"몇 달 전에 그거 왜 그렇게 짰더라?" 싶을 때, 흩어진 대화 기록을 한국어로 그냥 물어보면
관련된 과거 대화를 찾아 **근거(날짜)와 함께** 답해준다. 전부 로컬에서 돌고(외부 API·인터넷 불필요),
파이썬 의존성은 **`numpy` 하나**다.

```text
$ rag.py ask "작년에 그 캐시 TTL 왜 24시간으로 잡았더라?"

원본 데이터가 하루 한 번 갱신되기 때문입니다. 더 짧게 잡으면 변하지 않은
데이터를 불필요하게 다시 조회하고, 더 길게 잡으면 갱신 후 옛 값이 오래
노출될 수 있어 24시간으로 합의했습니다. (2025-01-15)

└ 참고 2건 · topic: caching · 유사도 0.74
```

## ✨ 특징

- **하이브리드 검색** — 벡터(의미) + BM25(어휘)를 `α`로 결합. 의미도, 정확한 식별자도 잡는다.
- **로컬 전용** — Ollama(bge-m3 + qwen2.5) + numpy. 외부 API·인터넷 0.
- **자동 백업** — macOS launchd 가 매일 CLI 대화를 수집·증분 인덱싱(멱등).
- **웹 UI + API** — FastAPI, 답변이 실시간으로 타이핑되는 스트리밍.
- **측정되는 품질** — `eval.py` 로 recall@k / MRR, `--sweep` 으로 벡터/하이브리드/BM25 비교.
- **지켜지는 코드** — pytest 31개 + ruff + mypy, GitHub Actions CI.

---

## 핵심 아이디어: 파인튜닝이 아니라 RAG

대화를 모델에 **학습시키는** 게 아니다. 파인튜닝은 사실을 헷갈리고, 새 대화마다 다시 학습해야 한다.
대신 **검색이 관련 기록을 찾아 모델에게 읽혀주고**, 모델은 그걸 한국어로 정리만 한다.

```text
대화 export / Claude Code 세션 ──import──▶ logs/*.jsonl ──ingest──▶ 벡터 인덱스 + BM25
                                                                          │
                  질문 ─┬─ 벡터 임베딩(코사인) ─┐                          │
                        └─ BM25(어휘) ─────────┴─ α 결합 → 상위 k ◀────────┘
                                                      │
                                          로컬 LLM ──▶ 근거(날짜) 인용 답변
```

> LLM은 기억하는 두뇌가 아니라, 찾아온 기록을 읽어주는 입이다.

- **임베딩**: Ollama `bge-m3` (다국어, 한국어 우수, 1024차원)
- **생성**: Ollama `qwen2.5:7b` (기본) — 환경변수 `CHATLOG_LLM`으로 교체
- **검색**: 하이브리드 — 벡터(의미) + BM25(어휘)를 `α`로 결합. numpy 행렬곱이라 청크가
  10만 개를 넘기 전엔 0.1초 안쪽, 벡터DB 불필요.

### 왜 하이브리드인가

임베딩은 의미가 비슷한 문장을 잘 찾지만, `EasyPaymentController`·에러코드·`filesort`
같은 **정확한 식별자**는 놓치기 쉽다. BM25(어휘 검색)가 그 토큰을 직접 매칭해 보완한다.
`--alpha` 로 비중을 조절한다(`1`=순수 벡터, `0`=순수 BM25, 기본 `0.5`). BM25는 외부
의존성 없이 직접 구현(`bm25.py`)했고, 효과는 `eval.py --alpha` 로 수치 비교할 수 있다.

---

## 설치

> **한 방 설치:** 새 PC에서 클론/압축해제 후 `./setup.sh` 하나면 Ollama 모델 받기 →
> venv 생성 → 의존성 설치 → (있으면) Claude Code 로그 인덱싱 → MCP 등록까지 끝난다.
> (`--no-mcp` / `--no-index` 로 단계 생략 가능. Ollama 자체 설치만 `brew install ollama` 선행.)
> 아래는 그 과정을 수동으로 하나씩 푼 것이다.

```bash
brew install ollama
brew services start ollama          # 백그라운드 서버

ollama pull bge-m3                  # 임베딩 (~1.2GB)
ollama pull qwen2.5:7b             # 생성 (~4.7GB)
# 더 좋은 한국어를 원하면(메모리 32GB+): ollama pull qwen2.5:14b

python3 -m venv .venv               # homebrew python은 PEP668로 전역설치 막힘
.venv/bin/pip install numpy
```

---

## 사용법

### 1. 대화를 로그로 변환

```bash
# Claude Code 세션 transcript (~/.claude/projects 에 자동으로 쌓여 있음)
.venv/bin/python import_logs.py claude-code ~/.claude/projects

# 또는 claude.ai 데이터 export / 클로드 메모리 폴더
.venv/bin/python import_logs.py claude-export ~/Downloads/conversations.json
.venv/bin/python import_logs.py memory <메모리 폴더>
```

도구 호출·결과 같은 노이즈는 버리고 사람·AI 텍스트만 추출한다.
`topic`은 세션의 작업 디렉토리 이름으로 자동 분류되고, 파일명은 세션 id 기준이라 **다시 돌려도 중복이 안 쌓인다(멱등)**.

### 2. 인덱스 생성/갱신 (증분)

```bash
.venv/bin/python rag.py ingest      # 새/바뀐 청크만 임베딩
```

### 3. 묻기

```bash
.venv/bin/python rag.py ask "그 API 멱등키 왜 24시간이었지?"
.venv/bin/python rag.py ask "..." --topic <주제> --since 2025-01-01 --k 8 --min-sim 0.5

# 더 똑똑한 답이 필요할 때
CHATLOG_LLM=qwen2.5:14b .venv/bin/python rag.py ask "..."
```

| 옵션 | 설명 |
|---|---|
| `--topic`   | 주제로 사전 필터(완전일치) |
| `--since` / `--until` | 기간 필터 (YYYY-MM-DD) |
| `--k`       | 검색 청크 수 (기본 6) |
| `--min-sim` | 이 유사도 미만 청크는 버림 (기본 0.45) |
| `--alpha`   | 하이브리드 비중 (1=벡터, 0=BM25, 기본 0.5) |

---

## 웹 UI / API

CLI 대신 브라우저에서 묻고 싶으면 FastAPI 서버를 띄운다(인덱스가 있어야 함).

```bash
pip install -e ".[web]"      # fastapi, uvicorn
python3 server.py            # http://127.0.0.1:8000
```

- `GET /` — 검색창 웹 UI. 답변이 **실시간 타이핑(SSE 스트리밍)** 으로 나오고, 출처를 함께 표시
- `GET /api/health` — 상태/청크 수/모델
- `POST /api/ask` — `{question, k?, alpha?, topic?, generate?}` → `{answer, sources}`.
  `generate:false` 면 LLM 생성을 건너뛰고 검색 결과만(빠름)
- `POST /api/ask/stream` — 위와 같되 SSE(`sources`→`token`*→`done`)로 스트리밍
- `POST /api/reload` — 로그 갱신 후 인덱스 다시 로드

인덱스와 BM25 는 서버 시작 시 한 번만 로드한다. Ollama 가 죽어 있으면 503 으로 응답한다.

## MCP 로 붙여 쓰기 (Claude Code / Desktop)

웹·CLI 대신 **Claude 채팅 안에서** 바로 과거 기록을 검색하고 싶으면 MCP 서버로 붙인다.
같은 `rag.py` 로직·같은 인덱스를 쓰므로 웹/CLI 와 동시에 공존한다(입구만 하나 더 생긴다).

```bash
pip install -e ".[mcp]"      # mcp
```

도구 3개를 노출한다:

- `search_chatlog` — 관련 기록(원문 청크)만 반환. **답 요약은 Claude 가** 하므로 빠르다(주력).
- `ask_chatlog` — 로컬 qwen2.5 가 직접 한국어 답을 생성(CLI/웹과 동일 동작).
- `reload_chatlog` — `rag.py ingest` 후 메모리 인덱스 재적재.

Claude Code(`~/.claude.json` 또는 프로젝트 `.mcp.json`) / Desktop 설정에 등록:

```jsonc
{
  "mcpServers": {
    "chatlog": {
      "command": "/절대경로/chat-log/.venv/bin/python",
      "args": ["/절대경로/chat-log/mcp_server.py"]
    }
  }
}
```

> `command` 는 `numpy`+`mcp` 가 깔린 venv 의 python 을, `args` 는 이 저장소의 `mcp_server.py`
> **절대경로**를 가리킨다. 인덱스가 있어야 하고(`rag.py ingest`), 임베딩에 Ollama 가 떠 있어야 한다.
> 받는 사람은 위 두 경로만 자기 환경에 맞게 바꾸면 바로 붙는다.

## Docker 로 한 번에

`ollama` + 앱을 함께 띄운다(첫 실행은 모델 ~6GB 다운로드).

```bash
docker compose up --build      # → http://localhost:8000
```

동봉 `examples/` 로 자동 인덱싱돼 바로 검색된다. 실데이터는 `logs/` 를 마운트해 `rag.py ingest`.

## 매일 자동 백업 (macOS launchd)

대화 세션이 쌓이는 대로 매일 자동 수집하려면 LaunchAgent를 등록한다.
`auto_update.sh`가 `import → ingest`를 순서대로 실행한다.

```bash
# 설치 — 현재 컴퓨터 경로를 자동으로 채워 등록한다 (매일 09:00)
./install_launchd.sh

# 즉시 1회 실행 / 로그 / 끄기
launchctl kickstart -k gui/$(id -u)/com.chatlog.daily
tail -f auto_update.log
launchctl bootout gui/$(id -u)/com.chatlog.daily
```

`install_launchd.sh` 가 `com.chatlog.daily.plist.template` 의 경로 placeholder 를
이 프로젝트의 실제 위치로 치환해 `~/Library/LaunchAgents/` 에 설치한다.
`auto_update.sh` 도 자기 위치를 스스로 찾으므로, 폴더를 어디에 두든 그대로 동작한다.

- import는 **세션 id 기준 고정 파일명으로 덮어쓰기(멱등)** 라 매일 돌려도 중복이 안 쌓인다.
- `import_logs.py claude-code --prune` 은 원본에서 사라진 세션의 로그도 정리(삭제 동기화)한다.
- ingest는 증분이라 새/바뀐 청크만 임베딩한다(보통 수십 초).
- 실행 시 Ollama 서버가 떠 있어야 한다(`brew services start ollama`가 로그인 시 자동 기동).

> 자동 수집 대상은 **Claude Code(CLI) 세션뿐**이다. claude.ai 웹/앱 대화는 로컬에 없어 별도 export가 필요하다.

---

## 만들면서 배운 것 (RAG 함정 3종)

테스트 데이터 두 건만으로도 바로 드러난, 깔끔한 튜토리얼이 보여주지 않는 것들:

1. **무관한 기록이 답에 섞인다** → 상위 k개를 무조건 넣으면 관련 없는 청크까지 LLM이 사실인 양 끌어 쓴다.
   **유사도 임계값**으로 약한 근거는 버리고, 없으면 "찾지 못했다"로 환각을 차단.
2. **"작년"을 엉뚱한 해로 해석한다** → LLM은 오늘이 며칠인지 모른다.
   **프롬프트에 오늘 날짜를 주입**해 상대적 시간 표현을 풀게 한다.
3. **한국어로 물어도 다른 언어가 섞인다** → 다국어 모델의 고질병.
   **"반드시 한국어로만" 지시**를 프롬프트에 박는다.

그리고 — **export를 기다릴 필요가 없었다.** Claude Code는 이미 모든 세션을 로컬에 남기고 있었다.

---

## 테스트 & 평가

```bash
.venv/bin/pip install -e ".[dev]"   # pytest, ruff, mypy, fastapi, httpx

pytest          # 단위 테스트 31개 (Ollama 불필요 — 임베딩/생성을 가짜로 주입)
ruff check .    # 린트
mypy .          # 핵심 로직(rag.py, bm25.py) 타입 체크
```

테스트는 임베딩/생성을 가짜로 갈아끼워 **Ollama 없이** 청킹(턴 경계)·증분 인덱싱·하이브리드 검색·
메타 필터·임계값, importer 의 파싱·멱등성·삭제 동기화, FastAPI 엔드포인트(스트리밍 포함)까지
검증한다. push 마다 GitHub Actions 가 `ruff + mypy + pytest` 를 동일하게 돌린다.

### 검색 품질 측정 (recall@k / MRR)

검색이 "얼마나 잘 찾는지"를 숫자로 본다. 임계값 튜닝이나 임베딩 모델 교체 효과를 비교할 때 쓴다.

```bash
.venv/bin/python eval.py            # 기본 α 로 1회
.venv/bin/python eval.py --sweep    # 벡터/하이브리드/BM25 비교
```

```text
질문 10개 · 청크 5개 · 임베딩 bge-m3

모드                recall@1  recall@3  recall@5    MRR
───────────────────────────────────────────────────────
벡터                    1.00      1.00      1.00   1.000
하이브리드 α=0.5           1.00      1.00      1.00   1.000
BM25                  1.00      1.00      1.00   1.000
```

> 동봉한 `examples/` 는 주제가 뚜렷한 소규모 스모크 세트라 세 모드 모두 만점이다.
> 핵심은 점수가 아니라 **품질을 회귀(regression)로 잡고 α를 데이터로 고르는 틀** — 실제 로그
> (`--logs logs`)와 더 어려운 질문으로 돌리면 모드별·임계값별 트레이드오프가 숫자로 갈린다.

## 로그 포맷 (`logs/*.jsonl`)

한 줄 = 한 메시지. `examples/` 폴더에 샘플이 있다.

```json
{"date":"2025-01-15","source":"claude","topic":"caching","role":"user","text":"..."}
{"date":"2025-01-15","source":"claude","topic":"caching","role":"assistant","text":"..."}
```

| 필드 | 설명 |
|---|---|
| `date`   | 대화 날짜 (기간 필터에 사용) |
| `source` | claude-code / claude / gpt 등 출처 |
| `topic`  | 주제 태그 (주제 필터에 사용) |
| `role`   | user / assistant / memory |
| `text`   | 본문 |

`import_logs.py`가 위 형식으로 변환하며, 한 파일(=한 세션)을 **턴(메시지) 경계** 기준으로
~800자 청크로 묶어 임베딩한다(문장 중간을 자르지 않아 Q&A 쌍이 함께 검색된다).

---

## 확장 트리거

지금 구조로 충분하지만, 아래 시점엔 교체를 고려한다.

- **벡터DB(LanceDB 등)로 이전**: 인덱스 청크가 **10만 개 초과**하거나 `index.npz`가 메모리에 부담될 때.
  `embed()` / `_load_cache()`만 교체하면 된다(이미 분리되어 있음).
- **청크 품질**: 세션 요약을 함께 임베딩해 검색 정확도 향상.

---

## ⚠️ 개인정보 / 보안

`logs/`와 `index.npz`에는 **실제 대화 내용**이 담긴다. 업무·개인 정보가 섞일 수 있으므로
`.gitignore`가 이들을 커밋에서 제외한다. 저장소에는 **코드와 포맷 예시(`examples/`)만** 올라간다.
실제 데이터로 쓰는 디렉토리는 따로 두고, 공개 저장소엔 절대 로그를 올리지 않는다.

---

## 파일 구성

```text
rag.py            ingest(인덱스) · search(하이브리드 검색) · ask(질문)
bm25.py           경량 BM25 어휘 검색 (의존성 없음)
server.py         FastAPI 웹 서버 (스트리밍 UI + JSON API)
mcp_server.py     MCP 서버 (Claude Code/Desktop 에 붙이는 도구 3종)
setup.sh          새 PC 한 방 설치 (모델·venv·인덱싱·MCP 등록, 경로 자동 인식)
web/index.html    검색창 웹 UI (SSE 스트리밍, 자체 완결)
import_logs.py    claude-code / claude-export / memory → logs/*.jsonl (+ --prune)
eval.py           검색 품질 평가 (recall@k, MRR; --sweep)
auto_update.sh    매일 자동 수집 스크립트 (launchd가 실행, 경로 자동 인식)
install_launchd.sh                LaunchAgent 설치 (경로 자동 치환)
com.chatlog.daily.plist.template   LaunchAgent 템플릿
Dockerfile · docker-compose.yml · docker-entrypoint.sh   ollama+앱 한 번에
tests/            pytest 단위 테스트 31개 (Ollama 불필요)
eval/qa.jsonl     평가용 라벨 질문 세트
examples/         로그 포맷 예시 (평가 데이터 겸용)
pyproject.toml    의존성 · pytest/ruff/mypy 설정
.github/workflows/ci.yml          CI (ruff + mypy + pytest)
logs/             대화 로그 (gitignore — 커밋되지 않음)
```

## 스택

로컬 RAG · Ollama (bge-m3 + qwen2.5) · numpy · FastAPI · macOS launchd · Docker
