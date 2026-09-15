from __future__ import annotations

from dataclasses import dataclass, field
from app.config import BATCH_SIZE
from app.translator.parser import SubtitleLine


@dataclass
class Batch:
    batch_index: int
    lines: list[SubtitleLine]
    joined_text: str
    original_texts: list[str]

    @property
    def size(self) -> int:
        return len(self.lines)


@dataclass
class ContextBatcher:
    batch_size: int = BATCH_SIZE

    def create_batches(self, lines: list[SubtitleLine]) -> list[Batch]:
        if not lines:
            return []

        batches = []
        batch_idx = 0

        for i in range(0, len(lines), self.batch_size):
            batch_lines = lines[i:i + self.batch_size]
            original_texts = [l.text for l in batch_lines]
            joined = "\n".join(original_texts)

            batches.append(Batch(
                batch_index=batch_idx,
                lines=batch_lines,
                joined_text=joined,
                original_texts=original_texts,
            ))
            batch_idx += 1

        return batches

    def rejoin_translated(self, translated_batches: list[str], expected_count: int) -> list[str]:
        all_lines = []
        for batch_text in translated_batches:
            lines = [l.strip() for l in batch_text.strip().split("\n")]
            all_lines.extend(lines)

        while len(all_lines) < expected_count:
            all_lines.append("")

        return all_lines[:expected_count]
