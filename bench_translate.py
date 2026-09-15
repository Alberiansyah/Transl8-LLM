"""Benchmark: where does translation time actually go?"""
import sys
import io
import time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import httpx
from app.translator.llm_engine import build_system_prompt

BASE = "http://127.0.0.1:8080"


def timed_translate(texts, batch_label):
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    user_prompt = f"Translate these subtitle lines:\n\n{numbered}"
    system_prompt = build_system_prompt("en", "id", None)

    t0 = time.perf_counter()
    resp = httpx.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": "local-llm",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "top_p": 0.9,
            "reasoning_effort": "none",
        },
        timeout=300.0,
    )
    wall = time.perf_counter() - t0
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    timings = data.get("timings", {})
    usage = data.get("usage", {})
    print(f"--- {batch_label} ({len(texts)} lines) ---")
    print(f"  wall total:      {wall:.2f}s")
    print(f"  prompt:          {timings.get('prompt_n','?')} tok in {timings.get('prompt_ms',0)/1000:.2f}s ({timings.get('prompt_per_second','?')}/s)")
    print(f"  generated:       {timings.get('predicted_n','?')} tok in {timings.get('predicted_ms',0)/1000:.2f}s ({timings.get('predicted_per_second','?')}/s)")
    print(f"  -> generation = {(timings.get('predicted_ms',0)/1000)/wall*100:.0f}% of wall time")
    print(f"  output tokens:   {usage.get('completion_tokens','?')}")
    print(f"  finish_reason:   {data['choices'][0]['finish_reason']}")
    print(f"  content chars:   {len(content)}")
    print()


lines = [
    "Hey, did you hear about what happened to Sarah?",
    "No, what happened? Is she okay?",
    "She got that promotion she's been working toward for months.",
    "That's amazing! She totally deserves it.",
    "I know, right? She's been staying late every night this past quarter.",
    "Well, hard work pays off. I'm really happy for her.",
    "We should throw a party for her this weekend.",
    "Great idea! I'll talk to the others and arrange everything.",
    "Make sure to keep it a secret.",
    "Of course! Don't let it slip.",
]

# Batch A: 10 lines (typical single batch)
timed_translate(lines, "10-line batch x3 rounds")

# Same batch twice more to see prompt caching effect
for i in range(2):
    timed_translate(lines, f"10-line batch round {i+2} (cache?)")

# Batch B: 40 lines (bigger batch)
big = []
for i in range(40):
    big.append(f"Speaker {i%3} says something interesting about topic number {i}.")
timed_translate(big, "40-line batch")