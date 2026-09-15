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

# Translation defaults
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "15"))
MAX_LINE_LENGTH = int(os.getenv("MAX_LINE_LENGTH", "42"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))