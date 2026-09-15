# AGENTS.md

## Project

Subtitle Translator — context-aware subtitle translation web app powered by a **local LLM** (Qwen3.5-9B via llama-server's OpenAI-compatible API).

## Run

```bash
# 1. Start the local LLM server (llama.cpp) — add `-fa on` for flash attention (~1.5-2x faster)
.\llama-server.exe -m "E:\LLM\Qwen3.5-9B-Q8_0.gguf" --mmproj "E:\LLM\mmproj-Qwen3.5-9B-BF16.gguf" -ngl 99 -c 29000 -np 1 -fa on --host 127.0.0.1 --port 8080

# 2. Start the web app
python run.py
# → http://localhost:8000
```

## Architecture

- `app/main.py` — FastAPI app, mounts static files, serves index, calls `init_db()` on startup
- `app/config.py` — `LLM_BASE_URL` (llama-server endpoint), translation defaults (batch size, temperature, top_p, max_tokens), env vars
- `app/db.py` — SQLite layer: `init_db()`, `save_history()`, `get_history()`, `get_history_entry()`, `delete_history()`. DB file: `data.db` at project root.
- `app/api/routes.py` — REST endpoints (see API below)
- `app/api/models.py` — Pydantic models: `TranslateResponse`, `ProgressResponse`, `HistoryEntry`, `LanguageInfo`, `FileInfo`
- `app/translator/llm_engine.py` — Core translation engine: calls llama-server `/v1/chat/completions` via httpx. Builds numbered-line prompts, parses numbered translations back, includes glossary terms in the system prompt. Returns `list[str]`.
- `app/translator/pipeline.py` — Orchestrator: splits lines into context batches via `ContextBatcher`, calls `LLMEngine.translate_batch()` per batch, tracks `completed_lines`. Supports `cancel_event: threading.Event`. `TranslationPipeline.translate()` is the main entry.
- `app/translator/parser.py` — `load_subtitle()` / `save_subtitle()` using pysubs2. For ASS/SSA: extracts `{\...}` tags (prefix/suffix/unclosed) and restores them to translated output.
- `app/translator/context_batcher.py` — Groups `SubtitleLine`s into `Batch` objects. Non-overlapping. Each batch has `original_texts` list.
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
| `source_lang` | `en` | Source language code |
| `target_lang` | `id` | Target language code |
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
- The compact system prompt (~60 tokens) keeps prefill light; input lines dominate prefill time, not the prompt.

## Key Design Decisions

1. **Engine is a stateless HTTP client** — `llm_engine.py` makes synchronous httpx calls to the local llama-server. No model weights are loaded by the web app.
2. **Connection check endpoint** — `/api/llm-status` probes `GET {base}/v1/models`. The frontend shows a green "Connected: <model>" badge or a red "LLM Offline" badge.
3. **Prompt-based translation** — Each context batch is sent as one numbered list prompt (`1. ... 2. ...`). The LLM returns the same numbering; responses are parsed back into per-line translations. This leverages the LLM's large context window for dialogue continuity.
4. **Glossary via prompt** — Glossary terms are injected into the system prompt (`term => translation`) rather than placeholder substitution. The LLM is instructed to always use them.
5. **Graceful degradation** — On LLM timeout/connect error, the batch falls back to original text (logged). The job continues rather than failing.
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
- Local LLM: Qwen3.5-9B (Q8_0) through llama-server at `http://127.0.0.1:8080`
- SQLite (built-in, used for translation history)
- No torch/transformers dependency — translation is pure HTTP to the local LLM