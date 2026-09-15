from __future__ import annotations

import json
from pathlib import Path
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

    def apply(self, text: str) -> str:
        for entry in self.entries:
            if entry.case_sensitive:
                text = text.replace(entry.source, f"{{{{GL:{entry.target}}}}}")
            else:
                import re
                pattern = re.compile(re.escape(entry.source), re.IGNORECASE)
                text = pattern.sub(f"{{{{GL:{entry.target}}}}}", text)
        return text

    def restore(self, text: str) -> str:
        import re
        pattern = re.compile(r"\{\{GL:(.*?)\}\}")
        return pattern.sub(lambda m: m.group(1), text)

    def to_dict(self) -> list[dict]:
        return [{"source": e.source, "target": e.target, "case_sensitive": e.case_sensitive} for e in self.entries]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "Glossary":
        g = cls()
        for item in data:
            g.add(item["source"], item["target"], item.get("case_sensitive", True))
        return g

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Glossary":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)
