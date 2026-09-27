from __future__ import annotations

from dataclasses import dataclass
from app.config import BATCH_SIZE
from app.translator.parser import SubtitleLine


@dataclass
class Batch:
    batch_index: int
    lines: list[SubtitleLine]
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

            batches.append(Batch(
                batch_index=batch_idx,
                lines=batch_lines,
                original_texts=original_texts,
            ))
            batch_idx += 1

        return batches
