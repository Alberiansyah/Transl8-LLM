# AGENTS.md

## Project

Subtitle Translator — context-aware subtitle translation web app powered by a **local LLM** (Qwen3.5-9B via llama-server's OpenAI-compatible API).

## Run

```bash
# 1. Start the local LLM server (llama.cpp) — tuned for this subtitle workload
#    (-np = slots; set LLM_CONCURRENCY to match. See README "LLM Server Tuning".)
.\llama-server.exe -m "E:\LLM\Qwen3.5-9B-Q6_K.gguf" -ngl 99 -c 8192 -np 3 -fa on ^
  -ctk q8_0 -ctv q8_0 --cache-reuse 256 --reasoning off --metrics ^
  -cb --host 127.0.0.1 --port 8080

# 2. Start the web app
python run.py
# → http://localhost:8000
# Windows: double-click run.bat instead — starts the app and auto-opens the browser
```

## Architecture

- `app/main.py` — FastAPI app, mounts static files, serves index, calls `init_db()` on startup
- `app/config.py` — `LLM_BASE_URL` (llama-server endpoint), translation defaults (batch size, temperature, top_p, max_tokens), request tuning (`LLM_TIMEOUT`, `LLM_CONNECT_TIMEOUT`, `LLM_MAX_RETRIES`, `LLM_CONCURRENCY`, `MAX_LINES_PER_BATCH`), and upload/job limits (`MAX_UPLOAD_BYTES`, `JOB_TTL_SECONDS`, `ALLOWED_EXTENSIONS`), all env-overridable
- `app/db.py` — SQLite layer: `init_db()`, `save_history()`, `get_history()`, `get_history_entry()`, `delete_history()`. DB file: `data.db` at project root.
- `app/api/routes.py` — REST endpoints (see API below)
- `app/api/models.py` — Pydantic models: `TranslateResponse`, `ProgressResponse`, `HistoryEntry`, `LanguageInfo`, `FileInfo`
- `app/translator/llm_engine.py` — Core translation engine: calls llama-server `/v1/chat/completions` via httpx. Builds numbered-line prompts, parses numbered translations back, includes glossary terms in the system prompt. Returns `list[str]`.
- `app/translator/pipeline.py` — Orchestrator: splits lines into context batches via `ContextBatcher`, calls `LLMEngine.translate_batch()` per batch, tracks `completed_lines`. Supports `cancel_event: threading.Event`. `translate()` accepts `concurrency` (defaults to `LLM_CONCURRENCY`); when `>1` it runs batches in a `ThreadPoolExecutor` and calls `on_progress` only from the main thread. `TranslationPipeline.translate()` is the main entry.
- `app/translator/parser.py` — `load_subtitle()` / `save_subtitle()` using pysubs2. For ASS/SSA: extracts `{\...}` tags (prefix/suffix/unclosed) and restores them. Encoding fallback chain (`utf-8-sig → utf-8 → cp1252 → latin-1`); saves atomically via temp file + `os.replace`.
- `app/translator/context_batcher.py` — Groups `SubtitleLine`s into `Batch` objects. Non-overlapping. Each batch has an `original_texts` list.
- `app/translator/glossary.py` — `Glossary` class. Glossary entries are serialized and passed to the LLM as prompt context (no placeholder substitution needed).
- `app/static/` — index.html, style.css (dark theme), app.js

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/translate` | POST | Translate single file (multipart form) |
| `/api/translate-batch` | POST | Translate multiple files (multipart form) |
| `/api/progress/{id}` | GET | Translation progress (poll every 800ms) |
| `/api/cancel/{id}` | POST | Cancel in-progress translation |
| `/api/download/{id}` | GET | Download single translated file |
| `/api/download-batch/{id}` | GET | Download batch as ZIP |
| `/api/languages` | GET | Supported language list |
| `/api/llm-status` | GET | Local LLM server connection + model info |
| `/api/history` | GET | Translation history (50 latest, newest first) |
| `/api/history/{id}/download` | GET | Re-download from history (file or ZIP) |
| `/api/history/{id}` | DELETE | Delete history entry + output files |

## Translation Form Fields (POST /api/translate & /api/translate-batch)

| Field | Default | Description |
|-------|---------|-------------|
| `source_lang` | `English` | Source language **name** (sent verbatim to the LLM prompt) |
| `target_lang` | `Indonesian` | Target language **name** (sent verbatim to the LLM prompt) |
| `batch_size` | `15` | Lines grouped per LLM call (context window) |
| `temperature` | `0.3` | LLM creativity (0.1 precise → 0.7 creative) |
| `top_p` | `0.9` | Token sampling diversity |
| `max_tokens` | `2048` | Max output tokens per batch call |
| `glossary` | `[]` | JSON array of `{source, target, case_sensitive}` |

## Performance Characteristics (measured)

- **Bottleneck is token generation speed, not batching** — Qwen3.5-9B Q8_0 generates ~34 tokens/s. Each translated subtitle line costs roughly 15-20 output tokens, so throughput is ~2-2.5 lines/s regardless of `batch_size`.
- **`batch_size` has almost no effect** — measured 2.5 lines/s at batch 15 vs 2.1 lines/s at batch 60 (larger batches are slightly *slower* due to context bloat). Keep the default 15.
- **Prompt format matters** — the engine requests **numbered output** (`1. text\n2. text`) not JSON arrays. JSON arrays add ~4-8 tokens/line overhead for zero quality gain. `_try_json_array` is kept as a Qwen fallback (it occasionally emits comma-less arrays anyway).
- **Flash attention (`-fa on`)** can speed up generation ~1.5-2x if your llama.cpp build supports it. Add it to the llama-server command line and restart. Note: the default is `auto` — on GPU builds it usually auto-enables, so the speedup may already apply.
- **Concurrency** — set `LLM_CONCURRENCY` (>1) and raise llama-server `-np` to match. Decode is memory-bandwidth-bound, so aggregate throughput commonly scales ~1.8-3x, not Nx. Do not oversubscribe (4 slots max on a consumer GPU). Measure on your GPU. Keep `-c` sized for the slot count (e.g. `-c 8192 -np 3`), not the old `-c 29000`.
- **KV quantization** — `-ctk q8_0 -ctv q8_0` halves KV VRAM with negligible quality loss; do not use q4 KV for translation.
- **Prompt caching** — `--cache-reuse 256` (caching is on by default). The engine keeps the system prompt byte-stable across batches; glossary entries are **sorted deterministically** for this reason, so do not inject per-batch content into the system prompt.
- **Thinking mode** — pass `--reasoning off` (or `--reasoning-budget 0`); the app also sends `reasoning_effort: "none"`. Qwen3 thinking tokens are pure waste here.
- The compact system prompt (~60 tokens) keeps prefill light; input lines dominate prefill time, not the prompt.

## Key Design Decisions

1. **Engine is a stateless HTTP client** — `llm_engine.py` makes synchronous httpx calls to the local llama-server. No model weights are loaded by the web app.
2. **Connection check endpoint** — `/api/llm-status` probes `GET {base}/v1/models`. The frontend shows a green "Connected: <model>" badge or a red "LLM Offline" badge.
3. **Prompt-based translation** — Each context batch is sent as one numbered list prompt (`1. ... 2. ...`). The LLM returns the same numbering; responses are parsed back into per-line translations. This leverages the LLM's large context window for dialogue continuity.
4. **Glossary via prompt** — Glossary terms are injected into the system prompt (`term => translation`) rather than placeholder substitution. The LLM is instructed to always use them.
5. **Graceful degradation** — On LLM timeout/connect error or a malformed response (missing `choices`/`content`), the batch falls back to original text (logged). A single retry (controlled by `LLM_MAX_RETRIES`) is attempted first. The job continues rather than failing. A reusable `httpx.Client` (thread-safe) is shared across concurrent batches.
6. **ASS tag preservation** — `parser.py:extract_tags()` extracts prefix/suffix/auto-close tags. `save_subtitle()` reconstructs: `prefix + translated + auto_close + suffix`.
7. **Cancel via threading.Event** — `cancel_event` is passed to `pipeline.translate()`. Checked between context batches.
8. **History in SQLite** — Auto-saved on translation completion. In-memory `jobs` dict for active translations, `data.db` for completed history. Stores LLM params (temperature, max_tokens, top_p).

## Common Tasks

### Add new API endpoint
1. Define Pydantic model in `app/api/models.py`
2. Add route in `app/api/routes.py`
3. Frontend: add fetch call in `app/static/app.js`

### Change translation defaults
Edit `TEMPERATURE`, `TOP_P`, `MAX_TOKENS`, `BATCH_SIZE` in `app/config.py` and the UI defaults in `app/static/index.html` / `app/static/app.js`.

### Add / edit the language list
Edit the `LANGUAGES` list of `(code, name)` tuples in `app/api/routes.py`. The dropdown value sent to the LLM is the **name**. `source_lang`/`target_lang` are validated against this whitelist, so a new entry takes effect automatically but is otherwise rejected.

### Point to a different LLM server
Set `LLM_BASE_URL` (e.g., `http://127.0.0.1:8080` or a remote OpenAI-compatible endpoint) and `LLM_MODEL_NAME` in `app/config.py` or via env vars.

### Debug translation issues
Check server logs — `llm_engine.py` logs each batch: `Translated X lines via LLM in Y.YYYs`. Verify the LLM server is running with `curl http://127.0.0.1:8080/v1/models`.

## Testing

No test framework set up. Manual testing:
1. Start llama-server, then start the web app
2. Confirm the badge shows "Connected"
3. Upload `sample_en.srt`, translate en→id
4. Check download output has correct line count and numbered-prompt parity
5. Verify history entry appears after completion

For ASS tag testing, create a `.ass` file with `{\i1}`, `{\b1}` tags and verify they survive translation.

## Environment

- Python 3.14
- Local LLM: Qwen3.5-9B (Q6_K recommended; Q8_0 works) through llama-server at `http://127.0.0.1:8080`
- SQLite (built-in, used for translation history)
- No torch/transformers dependency — translation is pure HTTP to the local LLM