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
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse

from app.config import (
    UPLOAD_DIR, OUTPUT_DIR, LLM_BASE_URL,
    MAX_UPLOAD_BYTES, JOB_TTL_SECONDS, ALLOWED_EXTENSIONS, MAX_LINES_PER_BATCH,
)
from app.api.models import (
    TranslateResponse, ProgressResponse,
    LanguageInfo, HistoryEntry,
)
from app.translator.parser import load_subtitle, save_subtitle
from app.translator.pipeline import TranslationPipeline, TranslationProgress
from app.translator.llm_engine import LLMEngine
from app.translator.context_batcher import ContextBatcher
from app.translator.glossary import Glossary
from app.db import save_history, get_history, get_history_entry, delete_history

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

jobs: dict[str, dict] = {}

# Reused for the cheap status probe only; the translation pipeline is per-job.
_status_engine = LLMEngine()

# Full language list (code, name). Names are sent to the LLM prompt verbatim,
# so any language the model understands can be used — no fixed code validation.
LANGUAGES: list[tuple[str, str]] = [
    # Most common first
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
    # Extended list (alphabetical by code)
    ("af", "Afrikaans"), ("az", "Azerbaijani"), ("be", "Belarusian"),
    ("bs", "Bosnian"), ("ca", "Catalan"), ("ceb", "Cebuano"),
    ("co", "Corsican"), ("cy", "Welsh"), ("eo", "Esperanto"),
    ("et", "Estonian"), ("eu", "Basque"), ("fy", "Frisian"),
    ("ga", "Irish"), ("gd", "Scottish Gaelic"), ("gl", "Galician"),
    ("gu", "Gujarati"), ("ha", "Hausa"), ("haw", "Hawaiian"),
    ("hmn", "Hmong"), ("ht", "Haitian Creole"), ("hy", "Armenian"),
    ("ig", "Igbo"), ("is", "Icelandic"), ("iu", "Inuktitut"),
    ("jv", "Javanese"), ("kk", "Kazakh"), ("kn", "Kannada"),
    ("ku", "Kurdish"), ("ky", "Kyrgyz"), ("la", "Latin"),
    ("lb", "Luxembourgish"), ("lt", "Lithuanian"), ("lv", "Latvian"),
    ("mg", "Malagasy"), ("mi", "Maori"), ("mk", "Macedonian"),
    ("mn", "Mongolian"), ("mr", "Marathi"), ("mt", "Maltese"),
    ("ny", "Chichewa"), ("or", "Odia"), ("pa", "Punjabi"),
    ("ps", "Pashto"), ("sd", "Sindhi"), ("sm", "Samoan"),
    ("sn", "Shona"), ("so", "Somali"), ("sq", "Albanian"),
    ("st", "Sesotho"), ("su", "Sundanese"), ("tg", "Tajik"),
    ("ts", "Tsonga"), ("tt", "Tatar"), ("ug", "Uyghur"),
    ("uz", "Uzbek"), ("xh", "Xhosa"), ("yi", "Yiddish"),
    ("yo", "Yoruba"), ("zu", "Zulu"),
]

LANG_NAMES = {code: name for code, name in LANGUAGES}
VALID_LANG_NAMES = {name for _, name in LANGUAGES}


# ---------------------------------------------------------------------------
# Validation / lifecycle helpers
# ---------------------------------------------------------------------------

def _validate_languages(source_lang: str, target_lang: str):
    if source_lang not in VALID_LANG_NAMES:
        raise HTTPException(400, f"Unsupported source_lang: {source_lang}")
    if target_lang not in VALID_LANG_NAMES:
        raise HTTPException(400, f"Unsupported target_lang: {target_lang}")


def _validate_params(batch_size: int, temperature: float, max_tokens: int, top_p: float):
    if not (1 <= batch_size <= MAX_LINES_PER_BATCH):
        raise HTTPException(400, f"batch_size must be between 1 and {MAX_LINES_PER_BATCH}")
    if not (0.0 <= temperature <= 2.0):
        raise HTTPException(400, "temperature must be between 0.0 and 2.0")
    if not (0.0 < top_p <= 1.0):
        raise HTTPException(400, "top_p must be greater than 0.0 and at most 1.0")
    if not (1 <= max_tokens <= 32768):
        raise HTTPException(400, "max_tokens must be between 1 and 32768")


def _parse_glossary(raw: str) -> Glossary | None:
    """Parse the glossary form field into a Glossary, raising 400 on malformed input."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(400, "Invalid glossary")
    if not isinstance(data, list):
        raise HTTPException(400, "Invalid glossary")
    for item in data:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("source"), str)
            or not isinstance(item.get("target"), str)
        ):
            raise HTTPException(400, "Invalid glossary")
    if not data:
        return None
    return Glossary.from_dict(data)


def _check_upload(filename: str, content: bytes) -> tuple[str, str]:
    """Validate an upload's extension/size. Returns (safe_filename, extension)."""
    safe_name = Path(filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}. Use SRT, ASS, SSA, or VTT.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES} bytes)")
    return safe_name, ext


def _cleanup_jobs():
    now = time.time()
    for jid, j in list(jobs.items()):
        prog = j.get("progress")
        st = prog.status if prog else ""
        if st not in ("completed", "failed", "cancelled"):
            continue
        if now - j.get("start_time", now) > JOB_TTL_SECONDS:
            jobs.pop(jid, None)


def _cleanup_loop():
    while True:
        time.sleep(300)
        try:
            _cleanup_jobs()
        except Exception:
            logger.exception("Job cleanup failed")


threading.Thread(target=_cleanup_loop, daemon=True, name="job-cleanup").start()


def _record_history(job_id: str, job: dict, files: list[dict], elapsed: float):
    """Build and persist a history entry for a finished job.

    ``files`` is a list of dicts each exposing ``original_filename`` and
    ``output_path`` (a single-element list is used for non-batch jobs).
    """
    total_lines = job["total_lines"]
    lps = round(total_lines / elapsed, 1) if elapsed > 0 else 0.0
    save_history({
        "job_id": job_id,
        "source_lang": job["source_lang"],
        "target_lang": job["target_lang"],
        "is_batch": 1 if job.get("is_batch") else 0,
        "filenames": json.dumps([f["original_filename"] for f in files]),
        "output_paths": json.dumps([f["output_path"] for f in files]),
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
        "created_at": job.get("started_at") or datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat(),
    })


@router.get("/languages", response_model=list[LanguageInfo])
async def get_languages():
    return [LanguageInfo(code=c, name=n) for c, n in LANGUAGES]


@router.get("/llm-status")
async def get_llm_status():
    """Check the local LLM server connection."""
    status = await run_in_threadpool(_status_engine.check_connection)
    return {
        "connected": status.get("connected", False),
        "model": status.get("model"),
        "base_url": LLM_BASE_URL,
        "error": status.get("error"),
    }


@router.post("/translate", response_model=TranslateResponse)
async def start_translation(
    file: UploadFile = File(...),
    source_lang: str = Form("English"),
    target_lang: str = Form("Indonesian"),
    batch_size: int = Form(15),
    temperature: float = Form(0.3),
    max_tokens: int = Form(2048),
    top_p: float = Form(0.9),
    glossary: str = Form("[]"),
):
    _cleanup_jobs()
    _validate_languages(source_lang, target_lang)
    _validate_params(batch_size, temperature, max_tokens, top_p)
    gl = _parse_glossary(glossary)

    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES} bytes)")
    content = await file.read()
    safe_name, ext = _check_upload(file.filename or "", content)

    job_id = uuid.uuid4().hex[:12]
    save_path = UPLOAD_DIR / f"{job_id}{ext}"
    output_path = OUTPUT_DIR / f"{job_id}_translated{ext}"

    await run_in_threadpool(save_path.write_bytes, content)

    try:
        sub_data = await run_in_threadpool(load_subtitle, save_path)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse subtitle: {e}")
    finally:
        save_path.unlink(missing_ok=True)

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
        "original_filename": safe_name,
        "cancel_event": threading.Event(),
        "start_time": time.time(),
        "started_at": datetime.now().isoformat(),
        "total_lines": len(sub_data.lines),
    }

    thread = threading.Thread(target=_run_translation, args=(job_id,), daemon=True)
    thread.start()

    return TranslateResponse(
        job_id=job_id,
        status="queued",
        message=f"Translation started: {safe_name} ({source_lang} -> {target_lang})",
    )


def _run_translation(job_id: str):
    job = jobs[job_id]
    pipeline = TranslationPipeline()

    def on_progress(progress: TranslationProgress):
        job["progress"] = progress

    try:
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
            job["progress"] = TranslationProgress(status="cancelled")
            return

        save_subtitle(job["sub_data"], final_texts, job["output_path"])

        elapsed = round(time.time() - job["start_time"], 2)
        _record_history(
            job_id,
            job,
            [{"original_filename": job["original_filename"], "output_path": job["output_path"]}],
            elapsed,
        )

    except Exception as e:
        logger.error(f"Translation job {job_id} failed: {e}")
        job["progress"] = TranslationProgress(status="failed", error=str(e))
    finally:
        pipeline.llm.close()
        job["sub_data"] = None


@router.get("/progress/{job_id}", response_model=ProgressResponse)
async def get_progress(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    # A job whose progress is missing (or whose sub_data has been dropped after
    # completion) must not crash this endpoint; downloads rely on output paths only.
    progress = job.get("progress") or TranslationProgress(status="idle")
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
    progress = job.get("progress") or TranslationProgress(status="idle")
    status = progress.status
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
    if job.get("is_batch"):
        raise HTTPException(400, "This is a batch job. Use /api/download-batch/{job_id}")
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
    source_lang: str = Form("English"),
    target_lang: str = Form("Indonesian"),
    batch_size: int = Form(15),
    temperature: float = Form(0.3),
    max_tokens: int = Form(2048),
    top_p: float = Form(0.9),
    glossary: str = Form("[]"),
):
    if len(files) < 1:
        raise HTTPException(400, "At least 1 file required")

    _cleanup_jobs()
    _validate_languages(source_lang, target_lang)
    _validate_params(batch_size, temperature, max_tokens, top_p)
    gl = _parse_glossary(glossary)

    job_id = uuid.uuid4().hex[:12]

    file_entries = []
    for f in files:
        if f.size is not None and f.size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES} bytes)")
        content = await f.read()
        safe_name, ext = _check_upload(f.filename or "", content)

        file_id = uuid.uuid4().hex[:12]
        save_path = UPLOAD_DIR / f"{file_id}{ext}"
        output_path = OUTPUT_DIR / f"{file_id}_translated{ext}"

        await run_in_threadpool(save_path.write_bytes, content)

        try:
            sub_data = await run_in_threadpool(load_subtitle, save_path)
        except Exception as e:
            raise HTTPException(400, f"Failed to parse {safe_name}: {e}")
        finally:
            save_path.unlink(missing_ok=True)

        file_entries.append({
            "original_filename": safe_name,
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
        "started_at": datetime.now().isoformat(),
        "total_lines": total_lines,
    }

    thread = threading.Thread(target=_run_batch_translation, args=(job_id,), daemon=True)
    thread.start()

    names = ", ".join(e["original_filename"] for e in file_entries[:3])
    if len(file_entries) > 3:
        names += f" +{len(file_entries) - 3} more"

    return TranslateResponse(
        job_id=job_id,
        status="queued",
        message=f"Batch translation started: {len(file_entries)} files ({names})",
    )


def _run_batch_translation(job_id: str):
    job = jobs[job_id]
    files = job["files"]
    total_files = len(files)
    cancel_event = job["cancel_event"]
    pipeline = TranslationPipeline()

    total_completed_lines = 0

    # Each file has its own batch count; precompute so overall progress is exact.
    batches_per_file = [
        len(ContextBatcher(batch_size=job["batch_size"]).create_batches(f["sub_data"].lines))
        for f in files
    ]
    total_batches = sum(batches_per_file) or 1

    try:
        for i, file_entry in enumerate(files):
            if cancel_event.is_set():
                job["progress"] = TranslationProgress(status="cancelled")
                return

            completed_before = sum(batches_per_file[:i])

            def on_progress(
                progress: TranslationProgress,
                file_idx=i,
                fname=file_entry["original_filename"],
                completed_before=completed_before,
            ):
                nonlocal total_completed_lines
                file_completed = progress.completed_lines
                overall_completed = total_completed_lines + file_completed
                completed_batches = min(
                    completed_before + progress.completed_batches, total_batches
                )
                job["progress"] = TranslationProgress(
                    total_batches=total_batches,
                    completed_batches=completed_batches,
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
            total_batches=total_batches,
            completed_batches=total_batches,
            total_files=total_files,
            completed_files=total_files,
            completed_lines=job["total_lines"],
        )

        elapsed = round(time.time() - job["start_time"], 2)
        _record_history(job_id, job, files, elapsed)
    finally:
        pipeline.llm.close()
        for file_entry in files:
            file_entry["sub_data"] = None


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
