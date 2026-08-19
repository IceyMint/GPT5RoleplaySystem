#!/usr/bin/env bash
# Verify every commit in the range builds and passes the suite independently.
set -u
cd "$(dirname "$0")/.."

BASE=471a7dc
START_REF=$(git rev-parse --abbrev-ref HEAD)

# __pycache__ is untracked now, but stale .pyc can still confuse checkouts.
clean_pyc() { find . -name '__pycache__' -type d -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true; }

fail=0
for c in $(git log --format=%h ${BASE}..HEAD | tac); do
    clean_pyc
    git checkout -q "$c" 2>/dev/null
    result=$(timeout 600 python -m pytest tests -q 2>&1 | tail -1)
    subject=$(git log --format=%s -1 "$c" | cut -c1-50)
    case "$result" in
        *failed*) status="FAIL"; fail=1 ;;
        *)        status="ok  " ;;
    esac
    printf "  [%s] %s  %-52s %s\n" "$status" "$c" "$subject" "$result"
done

clean_pyc
git checkout -q "$START_REF"
echo
if [ "$fail" -eq 0 ]; then
    echo "All commits pass independently."
else
    echo "SOME COMMITS FAIL IN ISOLATION -- history is not bisectable."
fi
exit $fail
