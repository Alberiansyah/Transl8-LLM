from __future__ import annotations

import os
import re
import pysubs2
from pathlib import Path
from dataclasses import dataclass, field


TAG_PATTERN = re.compile(r'\{\\[^}]*\}')

# Subtitle formats understood by pysubs2, keyed by lowercase extension.
FORMAT_MAP = {"srt": "srt", "ass": "ass", "ssa": "ass", "vtt": "vtt"}

# pysubs2 SAVE identifiers keyed by lowercase extension. .ssa must be saved as
# SSA (not ASS) so the output keeps SSA syntax (v4.00 / [V4 Styles]).
SAVE_FORMAT_MAP = {"srt": "srt", "ass": "ass", "ssa": "ssa", "vtt": "vtt"}

OPEN_CLOSE_PAIRS = {
    "i": ("{\\i1}", "{\\i0}"),
    "b": ("{\\b1}", "{\\b0}"),
    "u": ("{\\u1}", "{\\u0}"),
    "s": ("{\\s1}", "{\\s0}"),
}

CLOSE_PATTERN = re.compile(r'\{\\([ibus])0\}')
OPEN_PATTERN = re.compile(r'\{\\([ibus])1\}')


def extract_tags(raw: str) -> tuple[str, str, str, str]:
    """Extract ASS/SSA tags from raw text.

    Returns:
        (prefix_tags, clean_text, suffix_tags, auto_close_tags)
    """
    all_tags = TAG_PATTERN.findall(raw)

    if not all_tags:
        return "", raw.strip(), "", ""

    visible_chars = TAG_PATTERN.sub('', raw)

    # Prefix tags = tags that appear before any visible (non-whitespace) text.
    prefix_tags = ""
    if visible_chars.strip():
        for tag in all_tags:
            tag_pos = raw.index(tag)
            before_tag = TAG_PATTERN.sub('', raw[:tag_pos])
            if re.search(r'[^\s\\]', before_tag):
                break
            prefix_tags += tag

    # Suffix tags = tags that appear after the last visible text character.
    suffix_tags = ""
    if visible_chars.strip():
        # Scan for the last non-whitespace character OUTSIDE any tag, so the
        # digits/letters inside tags (e.g. the "0" in {\i0}) are not counted.
        last_visible_idx = -1
        pos = 0
        while pos < len(raw):
            m = TAG_PATTERN.match(raw, pos)
            if m:
                pos = m.end()
                continue
            if not raw[pos].isspace():
                last_visible_idx = pos
            pos += 1

        for tag in reversed(all_tags):
            tag_pos = raw.index(tag)
            if tag_pos <= last_visible_idx:
                break
            between = TAG_PATTERN.sub('', raw[last_visible_idx + 1:tag_pos])
            if re.search(r'[^\s\\]', between):
                break
            suffix_tags = tag + suffix_tags

    clean = TAG_PATTERN.sub('', raw).strip()

    # Auto-close only tags that were opened but never explicitly closed.
    opens = {m.group(1) for m in OPEN_PATTERN.finditer(raw)}
    closes = {m.group(1) for m in CLOSE_PATTERN.finditer(raw)}

    auto_close = ""
    for tag_type in sorted(opens - closes):
        tag_pair = OPEN_CLOSE_PAIRS.get(tag_type)
        if tag_pair:
            auto_close += tag_pair[1]

    return prefix_tags, clean, suffix_tags, auto_close


@dataclass
class SubtitleLine:
    index: int
    start: int
    end: int
    text: str
    raw_text: str
    style: str = "Default"
    prefix_tags: str = ""
    suffix_tags: str = ""
    auto_close_tags: str = ""


@dataclass
class SubtitleFile:
    path: str
    format: str
    encoding: str
    lines: list[SubtitleLine] = field(default_factory=list)
    original_subs: pysubs2.SSAFile | None = field(default=None, repr=False)

    @property
    def line_count(self) -> int:
        return len(self.lines)


def clean_text(text: str) -> str:
    cleaned = TAG_PATTERN.sub('', text)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    # Normalize pysubs2 line-break markers (\N) to spaces
    # so the LLM sees a single continuous line of text
    cleaned = cleaned.replace('\\N', ' ').replace('\n', ' ')
    return cleaned.strip()


def load_subtitle(file_path: str | Path) -> SubtitleFile:
    path = Path(file_path)
    ext = path.suffix.lower().lstrip(".")

    fmt = FORMAT_MAP.get(ext, ext)

    subs = None
    used_encoding = "utf-8"
    last_error: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            subs = pysubs2.load(str(path), encoding=enc)
            used_encoding = enc
            break
        except UnicodeError as e:
            last_error = e
    if subs is None:
        raise last_error  # type: ignore[misc]

    lines = []
    for i, event in enumerate(subs):
        if event.is_comment:
            continue
        raw = event.text

        if fmt in ("ass", "ssa"):
            prefix, cleaned, suffix, auto_close = extract_tags(raw)
        else:
            cleaned = clean_text(raw)
            prefix, suffix, auto_close = "", "", ""

        if not cleaned:
            continue

        lines.append(SubtitleLine(
            index=i,
            start=event.start,
            end=event.end,
            text=cleaned,
            raw_text=raw,
            style=event.style,
            prefix_tags=prefix,
            suffix_tags=suffix,
            auto_close_tags=auto_close,
        ))

    return SubtitleFile(
        path=str(path),
        format=fmt,
        encoding=used_encoding,
        lines=lines,
        original_subs=subs,
    )


def save_subtitle(sub_data: SubtitleFile, translated_lines: list[str], output_path: str | Path) -> Path:
    out = Path(output_path)
    fmt = out.suffix.lower().lstrip(".")
    pysubs_fmt = SAVE_FORMAT_MAP.get(fmt, fmt)

    subs = sub_data.original_subs
    if subs is None:
        raise ValueError("SubtitleFile has no parsed original data to save")

    line_idx = 0
    text_idx = 0
    for event in subs:
        if event.is_comment:
            continue
        raw = event.text
        if pysubs_fmt in ("ass", "ssa"):
            _, cleaned, _, _ = extract_tags(raw)
        else:
            cleaned = clean_text(raw)
        if not cleaned:
            continue
        if text_idx < len(translated_lines) and line_idx < len(sub_data.lines):
            new_text = translated_lines[text_idx]
            line = sub_data.lines[line_idx]
            if pysubs_fmt in ("ass", "ssa"):
                event.text = line.prefix_tags + new_text + line.auto_close_tags + line.suffix_tags
            else:
                event.text = new_text
            text_idx += 1
        line_idx += 1

    # Atomic write: save to a temp file in the same directory, then replace,
    # so a mid-write shutdown cannot leave a truncated/corrupt output.
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        # format_ is required because the temp file extension (.tmp) is not
        # a recognised subtitle extension.
        subs.save(str(tmp), encoding="utf-8", format_=pysubs_fmt)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return out
