import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Local LLM server (llama-server OpenAI-compatible API)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "local-llm")

# --- Upload / job lifecycle ---
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50 MB per file
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
ALLOWED_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt"}

# --- LLM request tuning ---
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", "5"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "1"))  # >1 needs llama-server -np >= this
MAX_LINES_PER_BATCH = 100

# Translation defaults
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "15"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
