from __future__ import annotations

import re
import json
import time
import logging
import httpx
from app.config import LLM_BASE_URL, LLM_MODEL_NAME

logger = logging.getLogger(__name__)


def _coerce_strings(data: list) -> list[str]:
    """Convert a JSON list to a list of stripped strings."""
    return [str(x).strip() for x in data if isinstance(x, (str, int, float))]

# Compact translation system prompt — every token counts at generation speed.
BASE_SYSTEM_PROMPT = """Translate subtitle lines. Rules:
- Return numbered translations matching the input numbering exactly.
- Each line: <number>. <translation>
- Keep concise and natural for subtitle display.
- Preserve meaning and tone.
- If a line is already in the target language, keep it as-is.
- Do NOT add explanations, notes, or any text outside the numbered lines.


"""


def build_system_prompt(source_lang: str, target_lang: str, glossary: list[dict] | None = None) -> str:
    """Build the system prompt for translation, including glossary if provided."""
    prompt = (
        BASE_SYSTEM_PROMPT
        + f"\nTranslate from {source_lang} to {target_lang}."
    )

    if glossary:
        terms = "\n".join(
            f"- {g['source']} => {g['target']}"
            for g in glossary
        )
        prompt += (
            "\n\nUse these glossary terms when translating (ALWAYS use the given translation,"
            " preserving capitalization where appropriate):\n"
            + terms
        )

    return prompt


class LLMEngine:
    """Translation engine using a local LLM via llama-server (OpenAI-compatible API)."""

    def __init__(self):
        self.base_url = LLM_BASE_URL
        self.model = LLM_MODEL_NAME
        self.connected = False
        self.server_model = None

    def check_connection(self) -> dict:
        """Check if llama-server is reachable and get model info."""
        try:
            resp = httpx.get(f"{self.base_url}/v1/models", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    self.server_model = models[0].get("id", "unknown")
                    self.connected = True
                    return {
                        "connected": True,
                        "model": self.server_model,
                        "base_url": self.base_url,
                    }
            self.connected = False
            return {"connected": False, "error": "No models loaded on server"}
        except Exception as e:
            self.connected = False
            return {"connected": False, "error": str(e)}

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        glossary: list[dict] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        top_p: float = 0.9,
    ) -> list[str]:
        """Translate a batch of subtitle lines using the LLM.

        Sends all lines in a single prompt so the LLM has full context
        for accurate, context-aware translation.
        """
        if not texts:
            return []

        # Filter out empty lines but track their positions
        valid_indices = [j for j, t in enumerate(texts) if t.strip()]
        if not valid_indices:
            return [""] * len(texts)

        valid_texts = [texts[j] for j in valid_indices]

        # Build numbered input
        numbered_input = "\n".join(f"{i+1}. {t}" for i, t in enumerate(valid_texts))

        user_prompt = f"Translate these subtitle lines:\n\n{numbered_input}"

        system_prompt = build_system_prompt(source_lang, target_lang, glossary)

        # Estimate output tokens: ~2 tokens per word + 2 tokens overhead per line
        estimated_tokens = sum(len(t.split()) for t in valid_texts) * 2 + len(valid_texts) * 2 + 32
        request_max_tokens = max(256, min(max_tokens, estimated_tokens))

        t0 = time.perf_counter()

        try:
            response = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": request_max_tokens,
                    "top_p": top_p,
                    # Qwen3.x default to thinking mode which consumes the whole
                    # token budget on reasoning_content (content stays empty).
                    "reasoning_effort": "none",
                },
                timeout=180.0,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            logger.error("LLM request timed out")
            return texts  # Return originals on failure
        except httpx.ConnectError:
            logger.error("Cannot connect to LLM server at %s", self.base_url)
            return texts
        except Exception as e:
            logger.error("LLM request failed: %s", e)
            return texts

        elapsed = time.perf_counter() - t0

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()

        # Parse translations (numbered primary, JSON fallback)
        translated = self._parse_response(content, len(valid_texts))

        # If _parse_response returned None (neither format matched), use line fallback
        if translated is None:
            translated = self._line_fallback(content, len(valid_texts))
        # If count mismatches, use line fallback
        elif len(translated) != len(valid_texts):
            logger.warning(
                "Parsed %d translations but expected %d, trying line fallback",
                len(translated), len(valid_texts),
            )
            translated = self._line_fallback(content, len(valid_texts))
            if len(translated) != len(valid_texts):
                logger.warning("Fallback failed, using partial results + originals")
                for i in range(len(valid_texts)):
                    if i >= len(translated):
                        translated.append(valid_texts[i])

        # Map back to full list
        results = [""] * len(texts)
        for idx, text in zip(valid_indices, translated):
            results[idx] = text

        logger.info(
            "Translated %d lines via LLM in %.3fs",
            len(valid_texts), elapsed,
        )
        return results

    def _parse_response(self, content: str, expected_count: int) -> list[str]:
        """Parse the LLM response. Tries numbered lines first (primary format),
        then JSON array (legacy/fallback). Trims extras or pads later."""
        # 1. Numbered lines (primary — matches prompt format)
        parsed = self._try_numbered(content)
        if len(parsed) == expected_count:
            return parsed
        # If LLM returned more lines than expected, trim to expected
        if len(parsed) > expected_count:
            return parsed[:expected_count]

        # 2. JSON array (Qwen sometimes returns this anyway)
        parsed = self._try_json_array(content)
        if parsed is not None:
            if len(parsed) == expected_count:
                return parsed
            if len(parsed) > expected_count:
                return parsed[:expected_count]

        # No exact match — return what we have (caller pads with originals)
        return parsed if parsed is not None else []

    def _try_json_array(self, content: str) -> list[str] | None:
        """Extract a JSON array of strings from the response.

        Handles strict JSON plus Qwen-style arrays that omit commas
        between string elements (e.g. ["a" "b" "c"]).
        """
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", content).strip()
        # Remove surrounding whitespace/text before the array
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return None
        array_text = match.group(0)

        # 1. Strict JSON parse
        try:
            data = json.loads(array_text)
            if isinstance(data, list):
                return _coerce_strings(data)
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Qwen-style: array of quoted strings with missing commas.
        #    Extract every quoted string inside the brackets in order.
        strings = re.findall(r'"((?:[^"\\]|\\.)*)"', array_text)
        if strings:
            return [s.encode().decode("unicode_escape", errors="replace").strip()
                    if "\\" in s else s.strip() for s in strings]

        return None

    def _try_numbered(self, content: str) -> list[str]:
        """Parse numbered translation response (e.g., '1. text\\n2. text')."""
        results = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^\d+[\.\)\:]\s*(.*)", line)
            if m:
                results.append(m.group(1).strip())
        return results

    def _line_fallback(self, content: str, expected_count: int) -> list[str]:
        """Last-resort: split by lines, strip numbering if present.
        Trims extras if too many; returns short list if too few (caller pads)."""
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if len(lines) != expected_count:
            # Strip stray numbers and JSON artifacts
            cleaned = []
            for line in lines:
                line = re.sub(r"^\d+[\.\)\:]\s*", "", line)
                line = line.strip('",[]')
                if line:
                    cleaned.append(line)
            lines = cleaned
        # If a single line contains all answers comma-separated, split it
        if len(lines) == 1 and expected_count > 1:
            parts = [p.strip() for p in lines[0].split(",") if p.strip()]
            if len(parts) == expected_count:
                return parts
        # Trim extras if too many; return short list if too few
        if len(lines) >= expected_count:
            return lines[:expected_count]
        return lines