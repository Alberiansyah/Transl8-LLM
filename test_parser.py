"""Test LLM engine parser against various response formats."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from app.translator.llm_engine import LLMEngine

e = LLMEngine()
passed = 0
total = 0


def check(name, result, expected_count):
    global passed, total
    total += 1
    if result is None:
        result = []
    ok = len(result) == expected_count
    status = "OK" if ok else "FAIL"
    print(f"  {name}: {status} (got {len(result)}, expected {expected_count}) -> {result}")
    if ok:
        passed += 1


print("=== JSON Array Parsing ===")
r = e._try_json_array('["Halo apa kabar?", "Saya baik"]')
check("clean JSON", r, 2)

r = e._try_json_array('```json\n["Satu", "Dua", "Tiga"]\n```')
check("markdown fence", r, 3)

r = e._try_json_array('Let me translate this.\n\n["Halo", "Dunia"]')
check("text prefix + JSON", r, 2)

r = e._try_json_array('Here are the translations:\n["مرحبا", "كيف حالك"]')
check("unicode JSON", r, 2)

r = e._try_json_array('[ "first", "second", "third" ]')
check("spaced JSON", r, 3)

r = e._try_json_array("some random text without json")
check("no JSON returns None", r, 0)  # should be None

print("\n=== Numbered Line Parsing ===")
r = e._try_numbered("1. Halo\n2. Dunia\n3. Apa kabar")
check("dot numbering", r, 3)

r = e._try_numbered("1) Halo\n2) Dunia")
check("paren numbering", r, 2)

r = e._try_numbered("1: Halo\n2: Dunia")
check("colon numbering", r, 2)

print("\n=== Line Fallback ===")
r = e._line_fallback("Halo\nDunia\nApa kabar", 3)
check("plain lines", r, 3)

r = e._line_fallback("Halo, Dunia, Apa kabar", 3)
check("comma-separated single line", r, 3)

print("\n=== Full _parse_response ===")
r = e._parse_response('["Halo", "Dunia"]', 2)
check("full parse JSON", r, 2)

r = e._parse_response("1. Halo\n2. Dunia", 2)
check("full parse numbered", r, 2)

# Qwen3.5 thinking mode output (actual likely format)
qwen_output = (
    "I'll translate these subtitle lines from English to Indonesian.\n\n"
    '["Hai, apa yang terjadi dengan Sarah?", '
    '"Tidak, apa yang terjadi? Apakah dia baik-baik saja?", '
    '"Dia dapat promosi yang sudah dia kerjakan selama berbulan-bulan."]'
)
r = e._parse_response(qwen_output, 3)
check("Qwen thinking+JSON", r, 3)

# Actual Qwen3.5 output: JSON array WITH missing commas between elements
qwen_nocomma = (
    '[\n  "Hei, kamu dengar apa yang terjadi pada Sarah?"\n'
    '  "Tidak, apa yang terjadi? Apakah dia baik-baik saja?"\n'
    '  "Dia mendapatkan promosi yang sudah dia usahakan selama berbulan-bulan."\n]'
)
r = e._try_json_array(qwen_nocomma)
check("Qwen missing-commas JSON", r, 3)

print(f"\n{'='*40}")
print(f"Results: {passed}/{total} passed")
if passed == total:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
