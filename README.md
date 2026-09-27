# Subtitle Translator (LLM)

Context-aware subtitle translation web app. Translates SRT/ASS/SSA/VTT files using your **local LLM** (Qwen3.5 via llama-server) — 100% offline, no cloud API.

Fork of the NLLB-based [Transl8](https://github.com/Alberiansyah/Transl8) — the heavy torch/transformers NLLB engine is replaced by a lightweight HTTP client to a local llama-server.

## Features

- **Local LLM via llama-server** — Qwen3.5-9B or any OpenAI-compatible local server
- **Context-aware translation** — batches of lines sent together so the LLM keeps dialogue continuity
- **Automatic pronoun resolution** — detects character names and dialogue flow to resolve "Dia" → He/She/They correctly
- **107 languages** — full dropdown list (no typing needed): English, Indonesian, Javanese, Catalan, Swahili, Zulu, and more. Language names are sent to the LLM prompt directly, so any language the model understands works.
- **LLM parameter control** — temperature, top-p, max-tokens
- **Multi-file batch upload** — translate multiple files in one go, download as ZIP
- **ASS/SSA tag preservation** — italic, bold, positioning tags survive translation
- **Auto-close tags** — unclosed `{\i1}` gets `{\i0}` appended automatically
- **Glossary** — define terms for consistent translation (injected into the prompt)
- **Cancel translation** — stop in-progress translation between batches
- **Translation history** — SQLite-backed, re-download or delete past translations
- **Progress tracking** — elapsed time, ETA, lines/s, batch progress
- **Settings guide** — collapsible guide explaining Temperature, Top-P, Max Tokens, Context Batch
- **LLM status badge** — live connection + model name indicator
- **Original filename** on download
- **Context fallback protection** — prevents LLM output truncation from dropping lines

## Prerequisites

A running llama-server with the model loaded. The command below is tuned for this
subtitle workload (short prompts, many small batches, optional concurrency):

```powershell
.\llama-server.exe `
  -m "E:\LLM\Qwen3.5-9B-Q6_K.gguf" `
  -ngl 99 -c 8192 -np 3 -fa on `
  -ctk q8_0 -ctv q8_0 --cache-reuse 256 --reasoning off --metrics `
  -cb --host 127.0.0.1 --port 8080
```

> For a single-request setup use `-c 4096 -np 1`; for concurrency set `-np` to the
> same value as the app's `LLM_CONCURRENCY` (see Environment Variables). See
> [LLM Server Tuning](#llm-server-tuning-performance) for what each flag does.

## LLM Server Tuning (performance)

Recommended command line for this app (from llama.cpp docs/benchmarks):

```powershell
.\llama-server.exe -m "E:\LLM\Qwen3.5-9B-Q6_K.gguf" -ngl 99 -c 8192 -np 3 -fa on `
  -ctk q8_0 -ctv q8_0 --cache-reuse 256 --reasoning off --metrics `
  -cb --host 127.0.0.1 --port 8080
```

| Flag | Why |
|------|-----|
| `-c 4096` / `-c 8192` | Context size. Each subtitle request is only ~400-1500 tokens, so `4096` fits a single slot and `8192` fits `-np 3`. The old `-c 29000` wasted ~3.5 GiB VRAM. With unified KV (`-kvu`, default) `-c` is the shared pool across slots. |
| `-np N` | Number of parallel slots (2-4). Set the app's `LLM_CONCURRENCY` to match. Decode is memory-bandwidth-bound — one weight read serves all sequences — so aggregate throughput typically scales ~1.8-3x. Do **not** oversubscribe (4 max on a consumer GPU). Measure on your GPU. |
| `-fa on` | Flash attention. Faster generation and required for quantized KV on many backends. llama-server's default is `auto`, which usually enables it on GPU builds. |
| `-ctk q8_0 -ctv q8_0` | Quantize the KV cache to half its VRAM with negligible quality loss (measured ΔPPL ≈ +0.002). Do **not** use q4 KV for translation. |
| `--cache-reuse 256` | Reuses shared prompt prefixes; prompt caching is on by default. The app keeps its system prompt byte-stable (glossary entries are sorted for exactly this reason). |
| `--reasoning off` | Disables Qwen3 thinking tokens — pure waste for translation. (Equivalent to `--reasoning-budget 0`.) The app also sends `reasoning_effort: "none"`. |
| `--metrics` | Enables `/metrics` and `/slots` for tokens/s and cache-hit observability. |
| `-cb` | Continuous batching — lets new requests join while others decode. |
| `--defrag-thold` | **Deprecated/removed — do not use.** |
| `--mmproj` | **Dropped** — this workload has no vision input; removing it saves VRAM and load time. |

**Quantization.** `Q8_0` is generally overkill. `Q6_K` is near-lossless (≈ +0.07% PPL)
and roughly 30-50% faster (memory-bandwidth bound) with ~2 GB less VRAM. `Q5_K_M`
(≈ +0.24% PPL) is the best speed/VRAM compromise. Test lower quants against the
numbered-output parser — instruction-following degrades before raw quality does.

**Speculative decoding (optional).** Modern flags are
`--spec-type draft-eagle3 -md <draft.gguf> --spec-draft-n-max 8`. EAGLE-3 drafts
measured 1.6-2.2x on Qwen3-8B for structured output; expect ~1.3-1.8x for translation.
The old `--draft` flags are removed. Measure on your GPU.

> **Thinking-mode note:** Qwen3 recommends *not* using greedy decoding in thinking
> mode. Non-thinking (`temperature 0.3`) is the correct mode for translation.

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

**Or double-click `run.bat`** (Windows) — starts the app and opens `http://localhost:8000` in your browser automatically.

Note: the llama-server must already be running (see Prerequisites) for the badge to show `Connected`.

Open `http://localhost:8000` — the status badge should show `Connected: <model>`.

## Configuration

### UI Settings

| Setting | Default | Options | Description |
|---------|---------|---------|-------------|
| LLM Server | — | — | Status badge (Connected / Offline) |
| Temperature | 0.3 (Balanced) | 0.1 / 0.3 / 0.7 | LLM creativity |
| Top-P | 0.9 | 0.8 / 0.9 / 1.0 | Token sampling diversity |
| Max Tokens | 2048 | 1024 / 2048 / 4096 | Max output tokens per batch call |
| Context Batch | 15 | 5–50 | Lines grouped for context |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://127.0.0.1:8080` | llama-server (OpenAI-compatible) endpoint |
| `LLM_MODEL_NAME` | `local-llm` | Model name sent in API requests |
| `LLM_TIMEOUT` | `120` | Per-request timeout (seconds) for generation calls |
| `LLM_CONNECT_TIMEOUT` | `5` | Connection timeout (seconds) for the HTTP client |
| `LLM_MAX_RETRIES` | `1` | Retries on malformed response / transient error |
| `LLM_CONCURRENCY` | `1` | Concurrent batches. Must match llama-server `-np` when `>1` |
| `BATCH_SIZE` | `15` | Default context batch size |
| `MAX_LINES_PER_BATCH` | `100` | Hard cap on lines per batch (guards context overrun) |
| `TEMPERATURE` | `0.3` | Default temperature |
| `TOP_P` | `0.9` | Default top-p |
| `MAX_TOKENS` | `2048` | Default max tokens |
| `MAX_UPLOAD_BYTES` | `52428800` (50 MB) | Max size per uploaded subtitle file |
| `JOB_TTL_SECONDS` | `3600` | Age before finished/abandoned jobs are cleaned up |
| `ALLOWED_EXTENSIONS` | `.srt .ass .ssa .vtt` | Accepted upload extensions |

Or edit `app/config.py` directly.

## Supported Formats

| Format | Extensions | Tag Support |
|--------|-----------|-------------|
| SubStation Alpha | `.ass`, `.ssa` | `{\i1}`, `{\b1}`, `{\pos}`, etc. |
| SubRip | `.srt` | Plain text |
| WebVTT | `.vtt` | Plain text |

## Bug Fixes Applied

- **`max_tokens` capping** — output token limit was incorrectly calculated from input text length, causing the LLM to stop generating mid-batch and drop lines. Now uses the full `max_tokens` budget.
- **Line misalignment on empty numbered lines** — `_try_numbered` dropped empty-numbered entries, shifting every later line by one and silently corrupting subtitles. It now parses by explicit number and preserves positions; empty entries keep the original line.
- **Stored XSS** — untrusted subtitle/history content was rendered unescaped in the UI; all dynamic output is now escaped.
- **Shared pipeline race** — a single module-level pipeline held per-job progress/state, leaking across concurrent jobs. Each job now gets a fresh pipeline; `translate()` resets progress.
- **Input validation** — `source_lang`/`target_lang` are validated against the language whitelist, numeric params are range-checked, and glossary JSON is parsed defensively (malformed items skipped, never raises).
- **Uploads / jobs leak** — upload size and extension are checked before parsing, uploaded files are deleted after parse, and a background TTL thread purges stale jobs (`JOB_TTL_SECONDS`).
- **ASS auto-close duplication** — explicitly closed tags re-emitted a duplicate `{\i0}{\i0}`; auto-close is now computed only for tags left open.
- **Encoding fallback** — `load_subtitle` no longer hardcodes UTF-8; it tries `utf-8-sig → utf-8 → cp1252 → latin-1` before failing.
- **Polling leak** — frontend progress polling now stops cleanly on completion/cancel/error instead of continuing in the background.
- **Atomic output writes** — `save_subtitle` writes to a temp file and `os.replace`s it, so a mid-write shutdown cannot corrupt the output.
- **Parser truncation handling** — `_try_numbered` handles empty/truncated translation lines without producing parsing artifacts.
- **`\N` marker normalization** — pysubs2 line-break markers (`\N`) are normalized to spaces before sending to the LLM.
- **System prompt** — explicit translation example and "ALWAYS translate" instruction prevents the model from returning original text.
- **Context fallback** — `_line_fallback` preserves whatever the LLM generated instead of overwriting valid translations with original text.
- **Dialogue context hint** — auto-detects character names and dialogue flow, injects context into the LLM prompt for accurate "Dia" → He/She resolution.

## Performance (measured)

- Throughput is **~2-2.5 lines/s** on Qwen3.5-9B Q8_0 (~34 tokens/s generation). Each line costs ~15-20 output tokens.
- **`batch_size` has almost no effect** — larger batches are slightly *slower* (context bloat). Keep the default 15.
- The engine requests **numbered output** (`1. text\n2. text`), not JSON arrays — saves ~4-8 tokens/line.
- **Concurrency** — set `LLM_CONCURRENCY` (>1) and raise llama-server `-np` to the same value. Decode is memory-bandwidth-bound (one weight read serves all sequences), so aggregate throughput commonly improves ~1.8-3x, not Nx. Do not oversubscribe (4 slots max on a consumer GPU). This **must be measured on your GPU** — actual scaling depends on VRAM bandwidth and KV size.
- Speed is bound by the model's token generation rate; use flash attention (`-fa on`), quantized KV (`-ctk q8_0 -ctv q8_0`) and a smaller/faster model (e.g. Qwen3.5-4B) for significantly faster translation.
- **Qwen3.5-2B-Q8_0** achieves ~97% translation accuracy on 985-line subtitle files.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/translate` | POST | Translate single file |
| `/api/translate-batch` | POST | Translate multiple files |
| `/api/progress/{id}` | GET | Translation progress (poll) |
| `/api/cancel/{id}` | POST | Cancel in-progress translation |
| `/api/download/{id}` | GET | Download single translated file |
| `/api/download-batch/{id}` | GET | Download batch as ZIP |
| `/api/languages` | GET | Full supported language list (107) |
| `/api/llm-status` | GET | Local LLM server connection + model info |
| `/api/history` | GET | Translation history (50 latest) |
| `/api/history/{id}/download` | GET | Re-download from history |
| `/api/history/{id}` | DELETE | Delete history entry + files |

## Project Structure

```
TranslateLLM/
  app/
    main.py              # FastAPI entry point, startup init
    config.py            # LLM server URL, translation defaults, request/upload/job limits
    db.py                # SQLite history layer (init, CRUD)
    api/
      routes.py          # REST endpoints
      models.py          # Pydantic models
    translator/
      llm_engine.py      # LLM translation engine (llama-server API) + context resolver integration
      pipeline.py        # Translation orchestrator + validation (sequential or concurrent)
      parser.py          # Subtitle file parser (SRT/ASS/VTT) with \N normalization
      context_resolver.py # Auto-detect character names and dialogue flow for pronoun resolution
      context_batcher.py # Context-aware batching
      glossary.py        # Custom glossary
    static/
      index.html         # Web UI
      style.css          # Dark theme
      app.js             # Frontend logic
  data.db                # SQLite database (auto-created)
  uploads/               # Uploaded subtitle files
  output/                # Translated output files
  run.py                 # python run.py → localhost:8000
  requirements.txt
  sample_en.srt          # Sample file for testing
```

## Tech Stack

- **LLM**: [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server` (Qwen3.5) — OpenAI-compatible chat completions API
- **HTTP client**: httpx
- **Backend**: FastAPI + uvicorn
- **Subtitle parsing**: pysubs2
- **History**: SQLite (built-in, no extra dependency)

## License

MIT
