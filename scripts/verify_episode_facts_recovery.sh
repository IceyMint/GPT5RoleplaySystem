#!/usr/bin/env bash
# Prove the D/F regression tests catch the original defects by re-introducing them.
set -u
cd "$(dirname "$0")/.."

SRC=src/gpt5_roleplay_system
TEST=tests/test_episode_and_facts_recovery.py
cp $SRC/pipeline.py /tmp/fix_pipe_df.py
cp $SRC/memory.py   /tmp/fix_mem_df.py
restore() { cp /tmp/fix_pipe_df.py $SRC/pipeline.py; cp /tmp/fix_mem_df.py $SRC/memory.py; }
trap restore EXIT

run() { python -m pytest $TEST -q 2>/dev/null | tail -1; }

echo "=== 0. Both fixes in place (expect: all pass) ==="
run

echo
echo "=== 1. Bug D reintroduced: trim the live buffer by position ==="
python - <<'PY'
import pathlib
p = pathlib.Path("src/gpt5_roleplay_system/pipeline.py"); s = p.read_text()
s = s.replace("        retained = self._rolling_buffer.trim_after_episode(snapshot, overlap)",
              "        self._rolling_buffer.trim_to_last(overlap)\n        retained = overlap", 1)
p.write_text(s)
PY
run
restore

echo
echo "=== 2. Bug F reintroduced: drop evidence before the LLM call, no requeue ==="
python - <<'PY'
import pathlib, re
p = pathlib.Path("src/gpt5_roleplay_system/pipeline.py"); s = p.read_text()
# revert to: slice first, and let the exception escape instead of requeueing
s = s.replace("""                    self._requeue_fact_evidence(remaining, pending_participants)
                    self._facts_retry_not_before_ts = time.time() + FACTS_RETRY_COOLDOWN_SECONDS
                    return""",
"""                    raise""", 1)
s = s.replace("""                # Only drop the chunk from the work list once it actually succeeded.
                remaining = remaining[self._facts_evidence_max_messages :]""",
"""                pass""", 1)
s = s.replace("                chunk = remaining[: self._facts_evidence_max_messages]",
              "                chunk = remaining[: self._facts_evidence_max_messages]\n                remaining = remaining[self._facts_evidence_max_messages :]", 1)
p.write_text(s)
PY
run
restore

echo
echo "=== 3. Restored (expect: all pass again) ==="
run
