#!/usr/bin/env bash
# Verify each regression test actually catches the bug it was written for,
# by re-introducing the original buggy code one fix at a time.
set -u
cd "$(dirname "$0")/.."

SRC=src/gpt5_roleplay_system
TEST="tests/test_overflow_summary_regression.py tests/test_llm.py"
cp $SRC/pipeline.py /tmp/fix_pipeline.py
cp $SRC/memory.py   /tmp/fix_memory.py
cp $SRC/llm.py      /tmp/fix_llm.py

restore() { cp /tmp/fix_pipeline.py $SRC/pipeline.py; cp /tmp/fix_memory.py $SRC/memory.py; cp /tmp/fix_llm.py $SRC/llm.py; }
trap restore EXIT

run() { python -m pytest $TEST -q 2>/dev/null | tail -1; }

echo "=== 0. All fixes in place (expect: all pass) ==="
run

echo
echo "=== 1. Re-introduce llm.py bug (echo old summary on failure) ==="
python - <<'PY'
import pathlib
p = pathlib.Path("src/gpt5_roleplay_system/llm.py"); s = p.read_text()
s = s.replace('return payload.strip() if payload else ""', 'return payload.strip() if payload else summary', 1)
p.write_text(s)
PY
run
restore

echo
echo "=== 2. Re-introduce pipeline bug (apply summary on any truthy result) ==="
python - <<'PY'
import pathlib, re
p = pathlib.Path("src/gpt5_roleplay_system/pipeline.py"); s = p.read_text()
s = s.replace("""        cleaned = (summary_text or "").strip()
        if not cleaned or cleaned == previous_summary:""",
"""        cleaned = (summary_text or "").strip()
        if False:""", 1)
p.write_text(s)
PY
run
restore

echo
echo "=== 3. Re-introduce memory.py bug (cap overflow, keeping NEWEST) ==="
python - <<'PY'
import pathlib
p = pathlib.Path("src/gpt5_roleplay_system/memory.py"); s = p.read_text()
s = s.replace("previous_len = len(self._overflow)\n        self._overflow.extend(overflow)", "previous_len = len(self._overflow)\n        self._overflow.extend(overflow)\n        self._overflow = self._overflow[-OVERFLOW_WARN_THRESHOLD:]", 1)
p.write_text(s)
PY
run
restore

echo
echo "=== 4. Restored (expect: all pass again) ==="
run
