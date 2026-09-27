from __future__ import annotations

import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from dataclasses import dataclass

from app.config import LLM_CONCURRENCY
from app.translator.parser import SubtitleFile
from app.translator.context_batcher import ContextBatcher
from app.translator.glossary import Glossary
from app.translator.llm_engine import LLMEngine

logger = logging.getLogger(__name__)


@dataclass
class TranslationProgress:
    total_batches: int = 0
    completed_batches: int = 0
    status: str = "idle"
    error: str | None = None
    last_batch_seconds: float = 0.0
    device_used: str = "local-llm"
    completed_lines: int = 0
    total_files: int = 1
    completed_files: int = 0
    current_file: str = ""

    @property
    def percent(self) -> float:
        if self.total_batches == 0:
            return 0
        return (self.completed_batches / self.total_batches) * 100


class TranslationPipeline:
    def __init__(self):
        self.llm = LLMEngine()
        self.batcher = ContextBatcher()
        self.progress = TranslationProgress()

    def translate(
        self,
        sub_file: SubtitleFile,
        source_lang: str,
        target_lang: str,
        glossary: Glossary | None = None,
        batch_size: int = 15,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        on_progress: Callable[[TranslationProgress], None] | None = None,
        cancel_event: threading.Event | None = None,
        concurrency: int | None = None,
    ) -> list[str]:
        # Reset per-job state so nothing leaks between jobs on a reused instance.
        self.progress = TranslationProgress()
        self.batcher.batch_size = batch_size

        concurrency = concurrency or LLM_CONCURRENCY
        concurrency = max(1, concurrency)

        lines = sub_file.lines
        if not lines:
            self.progress.status = "completed"
            if on_progress:
                on_progress(self.progress)
            return []

        original_texts = [l.text for l in lines]

        # Glossary entries are passed to the LLM as prompt context
        glossary_entries = glossary.to_dict() if glossary and glossary.entries else None

        self.progress.status = "translating"
        if on_progress:
            on_progress(self.progress)

        batches = self.batcher.create_batches(lines)
        self.progress.total_batches = len(batches)

        all_translated = []

        def _run_batch(batch) -> list[str]:
            return self.llm.translate_batch(
                batch.original_texts, source_lang, target_lang,
                glossary=glossary_entries,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

        if concurrency <= 1:
            for batch in batches:
                if cancel_event and cancel_event.is_set():
                    self.progress.status = "cancelled"
                    if on_progress:
                        on_progress(self.progress)
                    return original_texts[:len(all_translated)] + [""] * (len(original_texts) - len(all_translated))

                batch_start = time.perf_counter()
                try:
                    translated = _run_batch(batch)
                    all_translated.extend(translated)
                except Exception as e:
                    logger.error(f"Batch {batch.batch_index} failed: {e}")
                    all_translated.extend(batch.original_texts)

                batch_elapsed = time.perf_counter() - batch_start

                self.progress.completed_batches += 1
                self.progress.completed_lines = len(all_translated)
                self.progress.last_batch_seconds = round(batch_elapsed, 2)
                self.progress.device_used = "local-llm"

                if on_progress:
                    on_progress(self.progress)
        else:
            # Concurrent batches. Results are keyed by batch_index so ordering
            # is preserved. on_progress is only ever called from this thread.
            results_by_index: dict[int, list[str]] = {}
            batch_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_index = {}
                for batch in batches:
                    if cancel_event and cancel_event.is_set():
                        break
                    future_to_index[executor.submit(_run_batch, batch)] = batch
                for future in as_completed(future_to_index):
                    batch = future_to_index[future]
                    try:
                        results_by_index[batch.batch_index] = future.result()
                    except Exception as e:
                        logger.error(f"Batch {batch.batch_index} failed: {e}")
                        results_by_index[batch.batch_index] = list(batch.original_texts)

                    self.progress.completed_batches += 1
                    self.progress.completed_lines = sum(
                        len(v) for v in results_by_index.values()
                    )
                    self.progress.last_batch_seconds = round(
                        time.perf_counter() - batch_start, 2
                    )
                    self.progress.device_used = "local-llm"
                    if on_progress:
                        # Only reached from the main thread (as_completed loop).
                        on_progress(self.progress)

            for batch in batches:
                if batch.batch_index in results_by_index:
                    all_translated.extend(results_by_index[batch.batch_index])

            if cancel_event and cancel_event.is_set():
                # Stop here: partial results aligned to original length.
                self.progress.status = "cancelled"
                if on_progress:
                    on_progress(self.progress)
                while len(all_translated) < len(original_texts):
                    all_translated.append("")
                return all_translated[:len(original_texts)]

        while len(all_translated) < len(original_texts):
            all_translated.append("")
        all_translated = all_translated[:len(original_texts)]

        # completed_lines reflects full alignment after padding.
        self.progress.completed_lines = len(all_translated)

        # Validate: detect lines that weren't translated
        untranslated = [(i, orig, trans) for i, (orig, trans)
                        in enumerate(zip(original_texts, all_translated))
                        if orig.strip() and orig.strip() == trans.strip()]
        if untranslated:
            logger.warning(
                "%d/%d lines may not have been translated "
                "(still match original text)",
                len(untranslated), len(original_texts)
            )

        self.progress.status = "completed"
        if on_progress:
            on_progress(self.progress)

        return all_translated