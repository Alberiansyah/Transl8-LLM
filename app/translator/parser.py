from __future__ import annotations

import re
import pysubs2
from pathlib import Path
from dataclasses import dataclass, field


TAG_PATTERN = re.compile(r'\{\\[^}]*\}')
INLINE_TAG_RE = re.compile(r'\{\\[^}]*\}')

ASS_TAG_CLOSE_MAP = {
    "i": ("i1", "i0"),
    "b": ("b1", "b0"),
    "u": ("u1", "u0"),
    "s": ("s1", "s0"),
    "fn": None,
    "fs": None,
    "c": None,
    "1c": None,
    "2c": None,
    "3c": None,
    "4c": None,
    "alpha": None,
    "1a": None,
    "2a": None,
    "3a": None,
    "4a": None,
    "fscx": None,
    "fscy": None,
    "frx": None,
    "fry": None,
    "frz": None,
    "fax": None,
    "fay": None,
    "shad": None,
    "ord": None,
    "pos": None,
    "move": None,
    "clip": None,
    "iclip": None,
    "t": None,
    "an": None,
    "a": None,
    "rf": None,
    "k": None,
    "kf": None,
    "ko": None,
    "st": None,
    "pbo": None,
    "p": None,
    "q": None,
    "be": None,
    "blur": None,
    " bord": None,
    "xbord": None,
    "ybord": None,
    "shadow": None,
    "xshad": None,
    "yshad": None,
    "frz": None,
    "org": None,
    "lay": None,
}

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

    first_tag_pos = raw.index(all_tags[0]) if all_tags else len(raw)

    last_tag_end = 0
    for t in all_tags:
        pos = raw.rfind(t)
        end = pos + len(t)
        if end > last_tag_end:
            last_tag_end = end

    visible_chars = TAG_PATTERN.sub('', raw)

    prefix_tags = ""
    if all_tags and visible_chars.strip():
        first_visible = 0
        for ch in raw:
            if ch not in ('{', '\\') and not ch.isspace():
                break
            first_visible += 1

        for tag in all_tags:
            tag_pos = raw.index(tag)
            tag_end = tag_pos + len(tag)
            before_tag = raw[:tag_pos]
            has_visible_before = bool(re.search(r'[^\s{}\\]', before_tag))
            if not has_visible_before:
                prefix_tags += tag
            else:
                break

    suffix_tags = ""
    if all_tags and visible_chars.strip():
        last_visible_idx = -1
        for i in range(len(raw) - 1, -1, -1):
            if raw[i] not in ('{', '}', '\\') and not raw[i].isspace():
                last_visible_idx = i
                break

        for tag in reversed(all_tags):
            tag_pos = raw.index(tag)
            if tag_pos > last_visible_idx + 1:
                between = raw[last_visible_idx + 1:tag_pos]
                if not re.search(r'[^\s{}\\]', between):
                    suffix_tags = tag + suffix_tags
            else:
                break

    clean = TAG_PATTERN.sub('', raw).strip()

    opens = set()
    closes = set()
    for m in OPEN_PATTERN.finditer(raw):
        opens.add(m.group(1))
    for m in CLOSE_PATTERN.finditer(raw):
        closes.discard(m.group(1))

    auto_close = ""
    for tag_type in sorted(opens):
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
    original_subs: object = field(default=None, repr=False)

    @property
    def line_count(self) -> int:
        return len(self.lines)


def clean_text(text: str) -> str:
    cleaned = TAG_PATTERN.sub('', text)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    return cleaned.strip()


def load_subtitle(file_path: str | Path) -> SubtitleFile:
    path = Path(file_path)
    ext = path.suffix.lower().lstrip(".")

    format_map = {"srt": "srt", "ass": "ass", "ssa": "ass", "vtt": "vtt"}
    fmt = format_map.get(ext, ext)

    subs = pysubs2.load(str(path), encoding="utf-8")

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
        encoding="utf-8",
        lines=lines,
        original_subs=subs,
    )


def save_subtitle(sub_data: SubtitleFile, translated_lines: list[str], output_path: str | Path) -> Path:
    out = Path(output_path)
    fmt = out.suffix.lower().lstrip(".")
    format_map = {"srt": "srt", "ass": "ass", "ssa": "ass", "vtt": "vtt"}
    pysubs_fmt = format_map.get(fmt, fmt)

    subs = sub_data.original_subs

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

    subs.save(str(out), encoding="utf-8")
    return out
