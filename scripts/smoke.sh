#!/usr/bin/env bash
set -euo pipefail

API=${API:-http://localhost:8000}

say() { echo "[smoke] $*"; }

say "Health check at $API/health"
if ! curl -s -S -m 5 -f "$API/health" >/dev/null; then
  say "Health check FAILED"
  exit 2
fi
say "Health: OK"

say "Starting sample job"
JOB_JSON=$(curl -s -S -f -m 10 -X POST "$API/sample" || true)
JOB_ID=$(python3 - <<'PY'
import sys, json
try:
    s=sys.stdin.read().strip()
    print(json.loads(s).get("job_id",""))
except Exception:
    print("")
PY
<<<"$JOB_JSON")
if [[ -z "$JOB_ID" ]]; then
  say "Failed to parse job_id from /sample"
  exit 3
fi
say "Job: $JOB_ID"

say "Polling status (up to 120s)"
ATTEMPTS=120
COMPLETED=0
for i in $(seq 1 $ATTEMPTS); do
  S=$(curl -s -S "$API/status/$JOB_ID" || true)
  STATUS=$(python3 - <<'PY'
import sys, json
try:
    j=json.loads(sys.stdin.read()); print(j.get("status",""))
except Exception:
    print("")
PY
<<<"$S")
  STAGE=$(python3 - <<'PY'
import sys, json
try:
    j=json.loads(sys.stdin.read()); print(j.get("stage",""))
except Exception:
    print("")
PY
<<<"$S")
  PROG=$(python3 - <<'PY'
import sys, json
try:
    j=json.loads(sys.stdin.read()); print(j.get("progress",0))
except Exception:
    print("0")
PY
<<<"$S")
  say "[$i] status=$STATUS stage=$STAGE progress=$PROG"
  if [[ "$STATUS" == "COMPLETED" || "$STATUS" == "FAILED" || "$STATUS" == "CANCELLED" ]]; then
    [[ "$STATUS" == "COMPLETED" ]] && COMPLETED=1
    break
  fi
  sleep 1
done

say "Fetching result"
R=$(curl -s -S "$API/result/$JOB_ID" || true)
python3 - <<'PY'
import sys, json
try:
    j=json.loads(sys.stdin.read())
except Exception:
    j={}
eda=j.get("eda",{}); modeling=j.get("modeling",{}); explain=j.get("explain",{}); rep=j.get("report_html","")
best=modeling.get("best") or {}
primary=best.get("f1") or best.get("r2") or best.get("rmse")
print("eda_cols=", len(eda.get("columns",[])))
print("task=", modeling.get("task"))
print("best_name=", best.get("name"))
print("primary=", primary)
print("report_html_len=", len(rep) if isinstance(rep,str) else 0)
PY
<<<"$R"

exit $(( 0 == COMPLETED ))

