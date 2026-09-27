from __future__ import annotations

import re
import json
import time
import logging
import httpx
from app.config import (
    LLM_BASE_URL,
    LLM_MODEL_NAME,
    LLM_TIMEOUT,
    LLM_CONNECT_TIMEOUT,
    LLM_MAX_RETRIES,
    LLM_CONCURRENCY,
)
from app.translator.context_resolver import get_context_resolver

logger = logging.getLogger(__name__)


def _coerce_strings(data: list) -> list[str]:
    """Convert a JSON list to a list of stripped strings."""
    return [str(x).strip() for x in data if isinstance(x, (str, int, float))]

# Compact translation system prompt — every token counts at generation speed.
BASE_SYSTEM_PROMPT = """You are a professional subtitle translator. ALWAYS translate every line from source to target language. NEVER keep the original text.

Example of CORRECT output (from Indonesian to English):
Input:  1. Halo, apa kabar?
        2. Saya baikah, terima kasih.
Output: 1. Hello, how are you?
        2. I'm fine, thank you.

Rules:
- Return numbered translations matching the input numbering exactly.
- Each line: <number>. <translation>
- Keep concise and natural for subtitle display.
- Preserve meaning and tone.
- Do NOT add explanations, notes, or any text outside the numbered lines.
- Never repeat the original text — always produce the translated version.
- Even one word lines must be translated (e.g., "Oke" -> "Okay").

"""

# Matches a numbered output line: "1. text", "2) text", "3: text".
# The separator must be followed by whitespace or end-of-line, and the digit
# run is capped, so SRT-style timelines ("00:00:01,000 --> ...") and decimals
# ("3.5") are NOT treated as numbered translation lines.
_NUMBERED_RE = re.compile(r"^\s*(\d{1,4})\s*[\.\)\:](?:\s+(.*))?$")


def build_system_prompt(source_lang: str, target_lang: str, glossary: list[dict] | None = None) -> str:
    """Build the system prompt for translation, including glossary if provided."""
    prompt = (
        BASE_SYSTEM_PROMPT
        + f"\nTranslate from {source_lang} to {target_lang}."
    )

    if glossary:
        # Deterministic ordering keeps the system prompt byte-stable across
        # batches, which lets llama-server reuse its prompt-prefix cache.
        entries = sorted(
            (g for g in glossary if isinstance(g, dict)),
            key=lambda g: (str(g.get("source", "")), str(g.get("target", ""))),
        )
        terms = "\n".join(
            f"- {g.get('source', '')} => {g.get('target', '')}"
            for g in entries
        )
        if terms:
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
        # Single reusable client — httpx.Client is thread-safe, so concurrent
        # batches can share it. Keeps the connection pool warm across requests.
        self._client = httpx.Client(
            timeout=httpx.Timeout(LLM_TIMEOUT, connect=LLM_CONNECT_TIMEOUT),
            limits=httpx.Limits(
                max_keepalive_connections=max(4, LLM_CONCURRENCY * 2)
            ),
        )

    def close(self) -> None:
        """Close the underlying HTTP client and its connection pool."""
        try:
            self._client.close()
        except Exception:
            pass

    def check_connection(self) -> dict:
        """Check if llama-server is reachable and get model info."""
        try:
            resp = self._client.get(f"{self.base_url}/v1/models")
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

        # Inject pronoun resolution context (e.g., "Dia" → "He"/"She").
        # Kept out of the system prompt so the system prompt stays cache-stable.
        context_hint = ""
        resolver = get_context_resolver()
        if len(valid_texts) >= 3:
            context_hint = resolver.build_prompt_hint(valid_texts)

        # context_hint already carries its own leading blank line.
        user_prompt = f"Translate these subtitle lines:\n\n{numbered_input}{context_hint}"

        system_prompt = build_system_prompt(source_lang, target_lang, glossary)

        # Output tokens: translation output can be as long as the input,
        # plus numbering overhead. Use max_tokens directly so the LLM
        # never gets starved mid-batch (which drops lines).
        request_max_tokens = max_tokens

        payload = {
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
        }

        t0 = time.perf_counter()

        attempts = max(1, LLM_MAX_RETRIES + 1)
        content: str | None = None
        finish_reason = ""
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                response = self._client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                choices = result["choices"]
                message = choices[0]["message"]
                content = (message["content"] or "").strip()
                finish_reason = choices[0].get("finish_reason", "")
                last_error = None
                break
            except Exception as e:
                # Malformed response (KeyError/IndexError/ValueError) or a
                # transient transport error — retry, then degrade gracefully.
                last_error = e
                content = None
                if attempt + 1 < attempts:
                    logger.warning(
                        "LLM batch attempt %d/%d failed (%s: %s), retrying",
                        attempt + 1, attempts, type(e).__name__, e,
                    )

        elapsed = time.perf_counter() - t0

        if content is None:
            logger.error("LLM request failed after %d attempt(s): %s", attempts, last_error)
            return texts  # Return originals on failure

        # Parse translations (numbered primary, JSON fallback)
        translated = self._parse_response(content, len(valid_texts))

        # Detect truncation: if the LLM hit the token ceiling, the last
        # line may be a partial fragment. Blank it so the original is kept.
        if finish_reason == "length" and translated:
            last = translated[-1].strip()
            if len(last.split()) < 2 and len(last) < 20:
                logger.warning("Last translation line looks truncated, dropping it")
                translated[-1] = ""

        # Guard exact length (parse already pads, but stay defensive).
        if len(translated) < len(valid_texts):
            translated = translated + [""] * (len(valid_texts) - len(translated))
        elif len(translated) > len(valid_texts):
            translated = translated[:len(valid_texts)]

        # Map back to full list. Empty/whitespace entries keep the original
        # text for that specific line so alignment never drifts.
        results = [""] * len(texts)
        for idx, text in zip(valid_indices, translated):
            if text is None or not str(text).strip():
                results[idx] = texts[idx]   # untranslated -> keep original, preserves alignment
            else:
                results[idx] = text

        logger.info(
            "Translated %d lines via LLM in %.3fs",
            len(valid_texts), elapsed,
        )
        return results

    def _parse_response(self, content: str, expected_count: int) -> list[str]:
        """Parse the LLM response. Tries numbered lines first (primary format),
        then JSON array (legacy/fallback), then raw-line fallback.
        Always returns a list of length ``expected_count`` (padded with "")."""
        # 1. Numbered lines (primary — matches prompt format)
        parsed = self._try_numbered(content)
        if parsed:
            if len(parsed) < expected_count:
                parsed = parsed + [""] * (expected_count - len(parsed))
            elif len(parsed) > expected_count:
                parsed = parsed[:expected_count]
            return parsed

        # 2. JSON array (Qwen sometimes returns this anyway)
        parsed = self._try_json_array(content)
        if parsed:
            if len(parsed) == expected_count:
                return parsed
            if len(parsed) > expected_count:
                return parsed[:expected_count]
            return parsed + [""] * (expected_count - len(parsed))

        # 3. Last-resort raw line extraction
        parsed = self._line_fallback(content, expected_count)
        if len(parsed) < expected_count:
            parsed = parsed + [""] * (expected_count - len(parsed))
        elif len(parsed) > expected_count:
            parsed = parsed[:expected_count]
        return parsed

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
            results = []
            for s in strings:
                if "\\" in s:
                    # Properly unescape JSON escapes; never mangle valid
                    # non-ASCII text that merely contains a backslash.
                    try:
                        s = json.loads('"' + s + '"')
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                results.append(s.strip())
            return results

        return None

    def _try_numbered(self, content: str) -> list[str]:
        """Parse numbered translation response (e.g., '1. text\\n2. text').

        Parses by explicit number so empty entries do not shift every
        later line. Returns a list covering the numbering range, with ""
        for any missing/empty number. Empty list if no numbered lines are
        present. Absurd numbers are ignored to avoid huge allocations.
        """
        entries: dict[int, str] = {}
        for line in content.split("\n"):
            m = _NUMBERED_RE.match(line)
            if m:
                n = int(m.group(1))
                if n > 10000:          # ignore absurd numbers (hallucinated timestamps etc.)
                    continue
                entries[n] = (m.group(2) or "").strip()
        if not entries:
            return []
        # Detect 0-based numbering so line 0 is not silently dropped.
        base = 0 if 0 in entries else 1
        max_num = max(entries)
        size = max_num - base + 1
        out = [""] * size
        for n, txt in entries.items():
            idx = n - base
            if 0 <= idx < size:
                out[idx] = txt
        return out

    def _line_fallback(self, content: str, expected_count: int) -> list[str]:
        """Last-resort: extract all lines from LLM response.
        Preserves whatever the LLM generated (English, Indonesian, or mixed).
        Only pads remaining gaps with originals at the pipeline level.
        Never overwrites already-extracted translations."""
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if not lines:
            return []

        # If lines have numbering, extract the text after the number
        numbered = []
        for line in lines:
            m = _NUMBERED_RE.match(line)
            if m:
                text = (m.group(2) or "").strip()
                if text:
                    numbered.append(text)

        if numbered:
            # We got numbered lines — use them as-is, even if count differs
            if len(numbered) >= expected_count:
                return numbered[:expected_count]
            return numbered

        # No numbering found — try to extract raw text lines
        # Strip JSON artifacts and stray numbers
        cleaned = []
        for line in lines:
            line = re.sub(r"^\d+[\.\)\:]\s*", "", line)
            line = line.strip('",[]')
            if line:
                cleaned.append(line)

        if not cleaned:
            return []

        if len(cleaned) >= expected_count:
            return cleaned[:expected_count]
        return cleaned
