# Subtitle Translator (LLM)

Context-aware subtitle translation web app. Translates SRT/ASS/SSA/VTT files using your **local LLM** (Qwen3.5-9B via llama-server) — 100% offline, no cloud API.

Fork of the NLLB-based [Transl8](https://github.com/Alberiansyah/Transl8) — the heavy torch/transformers NLLB engine is replaced by a lightweight HTTP client to a local llama-server.

## Features

- **Local LLM via llama-server** — Qwen3.5-9B (or any OpenAI-compatible local server)
- **Context-aware translation** — batches of lines sent together so the LLM keeps dialogue continuity
- **Any language** — 48 common languages in the dropdown, plus an "Other…" option to type any language (Javanese, Catalan, Swahili, etc.) and the LLM translates it
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

## Prerequisites

A running llama-server with the model loaded. Example (match your setup):

```powershell
.\llama-server.exe `
  -m "E:\LLM\Qwen3.5-9B-Q8_0.gguf" `
  --mmproj "E:\LLM\mmproj-Qwen3.5-9B-BF16.gguf" `
  -ngl 99 -c 29000 -np 1 -fa on `
  --host 127.0.0.1 --port 8080
```

> `-fa on` enables flash attention (~1.5-2x faster generation on supported builds).
> Note: llama-server default is `auto`, so it may already be enabled on GPU builds.

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

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
| `BATCH_SIZE` | `15` | Default context batch size |
| `MAX_LINE_LENGTH` | `42` | Max characters per subtitle line |
| `TEMPERATURE` | `0.3` | Default temperature |
| `TOP_P` | `0.9` | Default top-p |
| `MAX_TOKENS` | `2048` | Default max tokens |

Or edit `app/config.py` directly.

## Supported Formats

| Format | Extensions | Tag Support |
|--------|-----------|-------------|
| SubStation Alpha | `.ass`, `.ssa` | `{\i1}`, `{\b1}`, `{\pos}`, etc. |
| SubRip | `.srt` | Plain text |
| WebVTT | `.vtt` | Plain text |

## Performance (measured)

- Throughput is **~2-2.5 lines/s** on Qwen3.5-9B Q8_0 (~34 tokens/s generation). Each line costs ~15-20 output tokens.
- **`batch_size` has almost no effect** — larger batches are slightly *slower* (context bloat). Keep the default 15.
- The engine requests **numbered output** (`1. text\n2. text`), not JSON arrays — saves ~4-8 tokens/line.
- Speed is bound by the model's token generation rate; use flash attention (`-fa on`) and a smaller/faster model (e.g. Qwen3.5-4B) for significantly faster translation.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/translate` | POST | Translate single file |
| `/api/translate-batch` | POST | Translate multiple files |
| `/api/progress/{id}` | GET | Translation progress (poll) |
| `/api/cancel/{id}` | POST | Cancel in-progress translation |
| `/api/download/{id}` | GET | Download single translated file |
| `/api/download-batch/{id}` | GET | Download batch as ZIP |
| `/api/languages` | GET | Supported language list |
| `/api/llm-status` | GET | Local LLM server connection + model info |
| `/api/history` | GET | Translation history (50 latest) |
| `/api/history/{id}/download` | GET | Re-download from history |
| `/api/history/{id}` | DELETE | Delete history entry + files |

## Project Structure

```
TranslateLLM/
  app/
    main.py              # FastAPI entry point, startup init
    config.py            # LLM server URL, translation defaults
    db.py                # SQLite history layer (init, CRUD)
    api/
      routes.py          # REST endpoints
      models.py          # Pydantic models
    translator/
      llm_engine.py      # LLM translation engine (llama-server API)
      pipeline.py        # Translation orchestrator
      parser.py          # Subtitle file parser (SRT/ASS/VTT)
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

- **LLM**: [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server` (Qwen3.5-9B) — OpenAI-compatible chat completions API
- **HTTP client**: httpx
- **Backend**: FastAPI + uvicorn
- **Subtitle parsing**: pysubs2
- **History**: SQLite (built-in, no extra dependency)

## License

MIT