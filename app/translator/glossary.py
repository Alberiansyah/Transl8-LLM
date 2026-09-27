from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GlossaryEntry:
    source: str
    target: str
    case_sensitive: bool = True


@dataclass
class Glossary:
    entries: list[GlossaryEntry] = field(default_factory=list)

    def add(self, source: str, target: str, case_sensitive: bool = True):
        self.entries.append(GlossaryEntry(source=source, target=target, case_sensitive=case_sensitive))

    def remove(self, source: str):
        self.entries = [e for e in self.entries if e.source != source]

    def to_dict(self) -> list[dict]:
        return [{"source": e.source, "target": e.target, "case_sensitive": e.case_sensitive} for e in self.entries]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "Glossary":
        """Build a Glossary defensively: non-list input or malformed items
        are skipped rather than raising."""
        g = cls()
        if not isinstance(data, list):
            return g
        for item in data:
            if not isinstance(item, dict):
                continue
            source = item.get("source")
            target = item.get("target")
            if not isinstance(source, str) or not isinstance(target, str):
                continue
            g.add(source, target, bool(item.get("case_sensitive", True)))
        return g
