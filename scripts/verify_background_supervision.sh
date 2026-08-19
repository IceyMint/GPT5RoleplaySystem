#!/usr/bin/env bash
# Prove the supervision tests catch the pre-fix behaviour by reverting the helper
# to a bare create_task (no strong ref, no error reporting).
set -u
cd "$(dirname "$0")/.."

SRC=src/gpt5_roleplay_system
TEST=tests/test_background_supervision.py
cp $SRC/background.py /tmp/fix_background.py
restore() { cp /tmp/fix_background.py $SRC/background.py; }
trap restore EXIT

run() { python -m pytest $TEST -q 2>/dev/null | tail -1; }

echo "=== 0. Supervision in place (expect: all pass) ==="
run

echo
echo "=== 1. Revert to bare create_task (old fire-and-forget behaviour) ==="
python - <<'PY'
import pathlib, re
p = pathlib.Path("src/gpt5_roleplay_system/background.py"); s = p.read_text()
start = s.index("    task = asyncio.ensure_future(coro)")
end = s.index("    return task", start)
s = s[:start] + "    task = asyncio.ensure_future(coro)\n" + s[end:]
p.write_text(s)
PY
run
restore

echo
echo "=== 2. Restored (expect: all pass again) ==="
run
