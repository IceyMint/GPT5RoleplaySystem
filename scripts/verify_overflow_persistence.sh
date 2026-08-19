#!/usr/bin/env bash
# Verify the overflow-persistence tests actually catch the original defect,
# by re-introducing the pre-fix behaviour one piece at a time.
set -u
cd "$(dirname "$0")/.."

SRC=src/gpt5_roleplay_system
TEST="tests/test_overflow_persistence.py tests/test_overflow_summary_regression.py"
cp $SRC/memory.py /tmp/fix_memory2.py
cp $SRC/session_state.py /tmp/fix_state2.py

restore() { cp /tmp/fix_memory2.py $SRC/memory.py; cp /tmp/fix_state2.py $SRC/session_state.py; }
trap restore EXIT

run() { python -m pytest $TEST -q 2>/dev/null | tail -1; }

echo "=== 0. Fixes in place (expect: all pass) ==="
run

echo
echo "=== 1. Original behaviour: restore() clears overflow, snapshot() omits it ==="
python - <<'PY'
import pathlib
p = pathlib.Path("src/gpt5_roleplay_system/memory.py"); s = p.read_text()
s = s.replace('            "overflow": list(self._overflow),\n', '', 1)
s = s.replace("self._overflow = list(overflow) if isinstance(overflow, list) else []",
              "self._overflow = []", 1)
p.write_text(s)
PY
run
restore

echo
echo "=== 2. snapshot() persists overflow but SessionStateStore not updated ==="
echo "    (the trap: raw MemoryItem objects are not JSON serializable)"
python - <<'PY'
import pathlib
p = pathlib.Path("src/gpt5_roleplay_system/session_state.py"); s = p.read_text()
s = s.replace('            memory["overflow"] = self._dump_items(memory.get("overflow", []))\n', '', 1)
s = s.replace('            memory["overflow"] = self._load_items(memory.get("overflow", []))\n', '', 1)
p.write_text(s)
PY
run
restore

echo
echo "=== 3. Restored (expect: all pass again) ==="
run
