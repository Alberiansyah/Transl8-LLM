from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


class TranslateRequest(BaseModel):
    source_lang: str = "English"
    target_lang: str = "Indonesian"
    batch_size: int = 15
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.9
    glossary: list[dict] = []


class GlossaryEntryModel(BaseModel):
    source: str
    target: str
    case_sensitive: bool = True


class TranslateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ProgressResponse(BaseModel):
    job_id: str
    status: str
    percent: float
    total_batches: int
    completed_batches: int
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    last_batch_seconds: float = 0.0
    device_used: str = "local-llm"
    lines_per_second: float = 0.0
    total_files: int = 1
    completed_files: int = 0
    current_file: str = ""
    total_lines: int = 0
    completed_lines: int = 0
    eta_seconds: float = 0.0


class LanguageInfo(BaseModel):
    code: str
    name: str


class FileInfo(BaseModel):
    filename: str
    format: str
    line_count: int
    detected_source: Optional[str] = None


class HistoryEntry(BaseModel):
    job_id: str
    source_lang: str
    target_lang: str
    is_batch: bool
    filenames: list[str]
    output_paths: list[str]
    total_lines: int = 0
    completed_lines: int = 0
    device: str = "local-llm"
    device_used: str = "local-llm"
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.9
    status: str = "completed"
    elapsed_seconds: float = 0.0
    lines_per_second: float = 0.0
    created_at: str = ""
    completed_at: str = ""