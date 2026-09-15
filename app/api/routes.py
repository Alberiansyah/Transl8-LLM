from __future__ import annotations

import io
import time
import uuid
import json
import zipfile
import threading
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from app.config import UPLOAD_DIR, OUTPUT_DIR, LLM_BASE_URL
from app.api.models import (
    TranslateResponse, ProgressResponse,
    LanguageInfo, FileInfo, HistoryEntry,
)
from app.translator.parser import load_subtitle, save_subtitle
from app.translator.pipeline import TranslationPipeline, TranslationProgress
from app.translator.glossary import Glossary
from app.db import save_history, get_history, get_history_entry, delete_history

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

jobs: dict[str, dict] = {}
pipeline = TranslationPipeline()

LANG_NAMES = {
    "en": "English", "id": "Indonesian", "ms": "Malay",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "th": "Thai", "vi": "Vietnamese", "tl": "Filipino",
    "ar": "Arabic", "hi": "Hindi", "bn": "Bengali",
    "pt": "Portuguese", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "ru": "Russian",
    "tr": "Turkish", "pl": "Polish", "nl": "Dutch",
    "sv": "Swedish", "no": "Norwegian", "da": "Danish",
    "fi": "Finnish", "cs": "Czech", "sk": "Slovak",
    "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian",
    "hr": "Croatian", "sr": "Serbian", "uk": "Ukrainian",
    "el": "Greek", "he": "Hebrew", "fa": "Persian",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "ml": "Malayalam", "my": "Myanmar", "km": "Khmer",
    "lo": "Lao", "ka": "Georgian", "am": "Amharic",
    "ne": "Nepali", "si": "Sinhala", "ur": "Urdu",
}


@router.get("/languages", response_model=list[LanguageInfo])
async def get_languages():
    common = [
        ("en", "English"), ("id", "Indonesian"), ("ms", "Malay"),
        ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Chinese"),
        ("th", "Thai"), ("vi", "Vietnamese"), ("tl", "Filipino"),
        ("ar", "Arabic"), ("hi", "Hindi"), ("bn", "Bengali"),
        ("pt", "Portuguese"), ("es", "Spanish"), ("fr", "French"),
        ("de", "German"), ("it", "Italian"), ("ru", "Russian"),
        ("tr", "Turkish"), ("pl", "Polish"), ("nl", "Dutch"),
        ("sv", "Swedish"), ("no", "Norwegian"), ("da", "Danish"),
        ("fi", "Finnish"), ("cs", "Czech"), ("sk", "Slovak"),
        ("hu", "Hungarian"), ("ro", "Romanian"), ("bg", "Bulgarian"),
        ("hr", "Croatian"), ("sr", "Serbian"), ("uk", "Ukrainian"),
        ("el", "Greek"), ("he", "Hebrew"), ("fa", "Persian"),
        ("sw", "Swahili"), ("ta", "Tamil"), ("te", "Telugu"),
        ("ml", "Malayalam"), ("my", "Myanmar"), ("km", "Khmer"),
        ("lo", "Lao"), ("ka", "Georgian"), ("am", "Amharic"),
        ("ne", "Nepali"), ("si", "Sinhala"), ("ur", "Urdu"),
    ]
    return [LanguageInfo(code=c, name=n) for c, n in common]


@router.get("/llm-status")
async def get_llm_status():
    """Check the local LLM server connection."""
    status = pipeline.llm.check_connection()
    return {
        "connected": status.get("connected", False),
        "model": status.get("model"),
        "base_url": LLM_BASE_URL,
        "error": status.get("error"),
    }


@router.post("/upload", response_model=FileInfo)
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in (".srt", ".ass", ".ssa", ".vtt"):
        raise HTTPException(400, f"Unsupported format: {ext}. Use SRT, ASS, SSA, or VTT.")

    file_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    content = await file.read()
    save_path.write_bytes(content)

    try:
        sub_data = load_subtitle(save_path)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse subtitle: {e}")

    return FileInfo(
        filename=file.filename,
        format=ext.lstrip("."),
        line_count=sub_data.line_count,
    )


@router.post("/translate", response_model=TranslateResponse)
async def start_translation(
    file: UploadFile = File(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("id"),
    batch_size: int = Form(15),
    temperature: float = Form(0.3),
    max_tokens: int = Form(2048),
    top_p: float = Form(0.9),
    glossary: str = Form("[]"),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in (".srt", ".ass", ".ssa", ".vtt"):
        raise HTTPException(400, f"Unsupported format: {ext}")

    job_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{job_id}{ext}"
    output_path = OUTPUT_DIR / f"{job_id}_translated{ext}"

    content = await file.read()
    save_path.write_bytes(content)

    try:
        sub_data = load_subtitle(save_path)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse subtitle: {e}")

    import json
    try:
        glossary_data = json.loads(glossary) if glossary else []
    except json.JSONDecodeError:
        glossary_data = []

    gl = Glossary.from_dict(glossary_data) if glossary_data else None

    jobs[job_id] = {
        "progress": TranslationProgress(status="queued"),
        "sub_data": sub_data,
        "output_path": str(output_path),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "batch_size": batch_size,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "glossary": gl,
        "original_filename": file.filename,
        "cancel_event": threading.Event(),
        "start_time": time.time(),
        "total_lines": len(sub_data.lines),
    }

    thread = threading.Thread(target=_run_translation, args=(job_id,), daemon=True)
    thread.start()

    return TranslateResponse(
        job_id=job_id,
        status="queued",
        message=f"Translation started: {file.filename} ({source_lang} -> {target_lang})",
    )


def _run_translation(job_id: str):
    job = jobs[job_id]

    def on_progress(progress: TranslationProgress):
        job["progress"] = progress

    try:
        start = time.time()
        final_texts = pipeline.translate(
            sub_file=job["sub_data"],
            source_lang=job["source_lang"],
            target_lang=job["target_lang"],
            glossary=job["glossary"],
            batch_size=job["batch_size"],
            temperature=job["temperature"],
            max_tokens=job["max_tokens"],
            top_p=job["top_p"],
            on_progress=on_progress,
            cancel_event=job["cancel_event"],
        )

        if job["cancel_event"].is_set():
            return

        save_subtitle(job["sub_data"], final_texts, job["output_path"])

        elapsed = round(time.time() - start, 2)
        total_lines = job["total_lines"]
        lps = round(total_lines / elapsed, 1) if elapsed > 0 else 0.0
        now = datetime.now().isoformat()

        save_history({
            "job_id": job_id,
            "source_lang": job["source_lang"],
            "target_lang": job["target_lang"],
            "is_batch": 0,
            "filenames": json.dumps([job["original_filename"]]),
            "output_paths": json.dumps([job["output_path"]]),
            "total_lines": total_lines,
            "completed_lines": total_lines,
            "device": "local-llm",
            "device_used": "local-llm",
            "temperature": job["temperature"],
            "max_tokens": job["max_tokens"],
            "top_p": job["top_p"],
            "status": "completed",
            "elapsed_seconds": elapsed,
            "lines_per_second": lps,
            "created_at": now,
            "completed_at": now,
        })

    except Exception as e:
        logger.error(f"Translation job {job_id} failed: {e}")
        job["progress"] = TranslationProgress(status="failed", error=str(e))


@router.get("/progress/{job_id}", response_model=ProgressResponse)
async def get_progress(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    progress = job["progress"]
    start_time = job.get("start_time", time.time())
    total_lines = job.get("total_lines", 0)

    elapsed = round(time.time() - start_time, 2) if progress.status not in ("idle", "queued") else 0.0

    completed_lines = progress.completed_lines
    if progress.status == "completed":
        completed_lines = total_lines

    lps = round(completed_lines / elapsed, 1) if elapsed > 0 and completed_lines > 0 else 0.0
    remaining = max(0, total_lines - completed_lines)
    eta = round(remaining / lps, 1) if lps > 0 and remaining > 0 else 0.0

    return ProgressResponse(
        job_id=job_id,
        status=progress.status,
        percent=progress.percent,
        total_batches=progress.total_batches,
        completed_batches=progress.completed_batches,
        error=progress.error,
        elapsed_seconds=elapsed,
        last_batch_seconds=progress.last_batch_seconds,
        device_used=progress.device_used,
        lines_per_second=lps,
        total_files=progress.total_files,
        completed_files=progress.completed_files,
        current_file=progress.current_file,
        total_lines=total_lines,
        completed_lines=completed_lines,
        eta_seconds=eta,
    )


@router.post("/cancel/{job_id}")
async def cancel_translation(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    status = job["progress"].status
    if status in ("completed", "failed", "cancelled"):
        raise HTTPException(400, f"Job already {status}")

    job["cancel_event"].set()
    logger.info(f"Cancellation requested for job {job_id}")
    return {"status": "cancelling", "job_id": job_id}


@router.get("/download/{job_id}")
async def download_translation(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    output_path = Path(job["output_path"])

    if not output_path.exists():
        raise HTTPException(404, "Translated file not ready yet")

    original_name = Path(job.get("original_filename", output_path.name)).stem
    ext = output_path.suffix

    return FileResponse(
        path=str(output_path),
        filename=f"{original_name}{ext}",
        media_type="application/octet-stream",
    )


@router.post("/translate-batch", response_model=TranslateResponse)
async def start_batch_translation(
    files: list[UploadFile] = File(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("id"),
    batch_size: int = Form(15),
    temperature: float = Form(0.3),
    max_tokens: int = Form(2048),
    top_p: float = Form(0.9),
    glossary: str = Form("[]"),
):
    if len(files) < 1:
        raise HTTPException(400, "At least 1 file required")

    job_id = str(uuid.uuid4())[:8]

    try:
        glossary_data = json.loads(glossary) if glossary else []
    except json.JSONDecodeError:
        glossary_data = []
    gl = Glossary.from_dict(glossary_data) if glossary_data else None

    file_entries = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in (".srt", ".ass", ".ssa", ".vtt"):
            raise HTTPException(400, f"Unsupported format: {f.filename}")

        file_id = str(uuid.uuid4())[:8]
        save_path = UPLOAD_DIR / f"{file_id}{ext}"
        output_path = OUTPUT_DIR / f"{file_id}_translated{ext}"

        content = await f.read()
        save_path.write_bytes(content)

        try:
            sub_data = load_subtitle(save_path)
        except Exception as e:
            raise HTTPException(400, f"Failed to parse {f.filename}: {e}")

        file_entries.append({
            "original_filename": f.filename,
            "sub_data": sub_data,
            "output_path": str(output_path),
        })

    total_lines = sum(e["sub_data"].line_count for e in file_entries)

    jobs[job_id] = {
        "progress": TranslationProgress(status="queued"),
        "files": file_entries,
        "current_file_index": 0,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "batch_size": batch_size,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "glossary": gl,
        "is_batch": True,
        "cancel_event": threading.Event(),
        "start_time": time.time(),
        "total_lines": total_lines,
    }

    thread = threading.Thread(target=_run_batch_translation, args=(job_id,), daemon=True)
    thread.start()

    names = ", ".join(f.filename for f in files[:3])
    if len(files) > 3:
        names += f" +{len(files) - 3} more"

    return TranslateResponse(
        job_id=job_id,
        status="queued",
        message=f"Batch translation started: {len(files)} files ({names})",
    )


def _run_batch_translation(job_id: str):
    job = jobs[job_id]
    files = job["files"]
    total_files = len(files)
    cancel_event = job["cancel_event"]

    total_completed_lines = 0

    for i, file_entry in enumerate(files):
        if cancel_event.is_set():
            job["progress"] = TranslationProgress(status="cancelled")
            return

        def on_progress(progress: TranslationProgress, file_idx=i, fname=file_entry["original_filename"]):
            nonlocal total_completed_lines
            file_completed = progress.completed_lines
            overall_completed = total_completed_lines + file_completed
            job["progress"] = TranslationProgress(
                total_batches=progress.total_batches * total_files,
                completed_batches=(file_idx * progress.total_batches) + progress.completed_batches,
                status=f"({file_idx + 1}/{total_files}) {fname}",
                last_batch_seconds=progress.last_batch_seconds,
                device_used=progress.device_used,
                completed_lines=overall_completed,
                total_files=total_files,
                completed_files=file_idx,
                current_file=fname,
            )

        try:
            final_texts = pipeline.translate(
                sub_file=file_entry["sub_data"],
                source_lang=job["source_lang"],
                target_lang=job["target_lang"],
                glossary=job["glossary"],
                batch_size=job["batch_size"],
                temperature=job["temperature"],
                max_tokens=job["max_tokens"],
                top_p=job["top_p"],
                on_progress=on_progress,
                cancel_event=cancel_event,
            )
            if cancel_event.is_set():
                job["progress"] = TranslationProgress(status="cancelled")
                return
            save_subtitle(file_entry["sub_data"], final_texts, file_entry["output_path"])
            total_completed_lines += len(file_entry["sub_data"].lines)
        except Exception as e:
            logger.error(f"Batch file {file_entry['original_filename']} failed: {e}")
            job["progress"] = TranslationProgress(status="failed", error=str(e))
            return

    job["progress"] = TranslationProgress(
        status="completed",
        total_batches=1,
        completed_batches=1,
        total_files=total_files,
        completed_files=total_files,
    )

    elapsed = round(time.time() - job["start_time"], 2)
    total_lines = job["total_lines"]
    lps = round(total_lines / elapsed, 1) if elapsed > 0 else 0.0
    now = datetime.now().isoformat()

    filenames = [fe["original_filename"] for fe in files]
    output_paths = [fe["output_path"] for fe in files]

    save_history({
        "job_id": job_id,
        "source_lang": job["source_lang"],
        "target_lang": job["target_lang"],
        "is_batch": 1,
        "filenames": json.dumps(filenames),
        "output_paths": json.dumps(output_paths),
        "total_lines": total_lines,
        "completed_lines": total_lines,
        "device": "local-llm",
        "device_used": "local-llm",
        "temperature": job["temperature"],
        "max_tokens": job["max_tokens"],
        "top_p": job["top_p"],
        "status": "completed",
        "elapsed_seconds": elapsed,
        "lines_per_second": lps,
        "created_at": now,
        "completed_at": now,
    })


@router.get("/download-batch/{job_id}")
async def download_batch(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if not job.get("is_batch"):
        raise HTTPException(400, "Not a batch job")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_entry in job["files"]:
            output_path = Path(file_entry["output_path"])
            if output_path.exists():
                original_name = Path(file_entry["original_filename"]).name
                zf.write(str(output_path), original_name)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=translated_{job_id}.zip"},
    )


@router.get("/history", response_model=list[HistoryEntry])
async def list_history():
    entries = get_history(limit=50)
    result = []
    for e in entries:
        result.append(HistoryEntry(
            job_id=e["job_id"],
            source_lang=e["source_lang"],
            target_lang=e["target_lang"],
            is_batch=bool(e["is_batch"]),
            filenames=e["filenames"],
            output_paths=e["output_paths"],
            total_lines=e["total_lines"],
            completed_lines=e["completed_lines"],
            device=e["device"],
            device_used=e["device_used"],
            temperature=e["temperature"],
            max_tokens=e["max_tokens"],
            top_p=e["top_p"],
            status=e["status"],
            elapsed_seconds=e["elapsed_seconds"],
            lines_per_second=e["lines_per_second"],
            created_at=e["created_at"],
            completed_at=e["completed_at"],
        ))
    return result


@router.get("/history/{job_id}/download")
async def download_history(job_id: str):
    entry = get_history_entry(job_id)
    if not entry:
        raise HTTPException(404, "History entry not found")

    output_paths = entry["output_paths"]
    filenames = entry["filenames"]

    if len(output_paths) == 1:
        p = Path(output_paths[0])
        if not p.exists():
            raise HTTPException(404, "Output file no longer exists on disk")
        orig_stem = Path(filenames[0]).stem if filenames else p.stem
        return FileResponse(
            path=str(p),
            filename=f"{orig_stem}{p.suffix}",
            media_type="application/octet-stream",
        )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for op, fn in zip(output_paths, filenames):
            p = Path(op)
            if p.exists():
                zf.write(str(p), Path(fn).name)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=translated_{job_id}.zip"},
    )


@router.delete("/history/{job_id}")
async def remove_history(job_id: str):
    entry = get_history_entry(job_id)
    if not entry:
        raise HTTPException(404, "History entry not found")

    for op in entry["output_paths"]:
        p = Path(op)
        if p.exists():
            p.unlink()

    delete_history(job_id)
    return {"status": "deleted", "job_id": job_id}