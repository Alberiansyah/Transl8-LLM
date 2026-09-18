"""Minimal context-aware pronoun resolution.

The code does NOT hardcode gender maps. It just extracts character names
and injects a brief context hint so the LLM can resolve "Dia" → He/She/They.
"""
from __future__ import annotations

import re

# Words that start with capital but aren't character names
NON_NAMES = {"diam", "dia", "itu", "apa", "siapa", "kenapa",
             "dimana", "mana", "bagaimana", "seperti", "karena",
             "oleh", "untuk", "dari", "dengan", "pada", "juga",
             "sudah", "baru", "yang", "ini", "atau", "tetapi",
             "tak", "tidak", "saya", "kau", "masih"}
COMMON_WORDS = {"the", "this", "that", "when", "where", "what", "who",
                "because", "before", "after", "but", "and", "or",
                "if", "then", "will", "would", "could", "should",
                "is", "are", "was", "were", "has", "have", "had",
                "not", "no", "yes", "can", "may", "might", "just",
                "been", "only", "also", "very", "much", "more",
                "most", "well", "here", "there", "still", "even"}


class ContextResolver:
    """Minimal context resolver — just extracts character names
    and passes them to the LLM prompt."""

    def build_prompt_hint(self, texts: list[str]) -> str:
        """Build a brief dialogue context hint for the LLM prompt.

        Extracts character names and 'Dia' references.
        Returns empty string if nothing useful found.
        """
        characters = set()
        dia_lines = []

        for idx, text in enumerate(texts):
            names = self._extract_names(text)
            for name in names:
                characters.add(name)
            if "dia" in text.lower():
                dia_lines.append(idx)

        if not characters and not dia_lines:
            return ""

        parts = []
        if characters:
            char_list = sorted(characters)[:8]
            parts.append(f"Characters: {', '.join(char_list)}")
        if dia_lines:
            parts.append(f"'Dia' appears in lines: {', '.join(str(i+1) for i in dia_lines[:5])}")
            parts.append("Note: 'Dia' = He/She/They depending on context. Read surrounding dialogue to determine.")

        return "\n\nDIALOGUE CONTEXT:\n" + "\n".join(f"  {p}" for p in parts)

    def _extract_names(self, text: str) -> list[str]:
        """Extract potential character names from a subtitle line."""
        words = re.findall(r'[A-Z][a-zA-Z]+', text)
        names = []
        for w in words:
            w_lower = w.lower()
            if len(w) < 2 or w_lower in NON_NAMES or w_lower in COMMON_WORDS:
                continue
            names.append(w)
        return names


# Module-level instance
_resolver: ContextResolver | None = None


def get_context_resolver() -> ContextResolver:
    """Get the shared context resolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = ContextResolver()
    return _resolver