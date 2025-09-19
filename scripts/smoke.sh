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

# OpenAI readiness checks
if [[ -n "${OPENAI_API_KEY:-}" || "${REQUIRE_OPENAI:-false}" == "true" ]]; then
  say "OpenAI non-live readiness check"
  OJ=$(curl -s -S -m 10 -f "$API/openai-smoke" || true)
  OOK=$(python3 - <<'PY'
import sys, json
try:
    j=json.loads(sys.stdin.read()); print(str(bool(j.get("ok", False))).lower())
except Exception:
    print("false")
PY
<<<"$OJ")
  if [[ "$OOK" != "true" ]]; then
    say "OpenAI non-live check FAILED"
    echo "[smoke] Response: $OJ"
    exit 4
  fi
  say "OpenAI non-live: OK"

  if [[ "${OPENAI_LIVE_SMOKE:-false}" == "true" ]]; then
    say "OpenAI LIVE check"
    OLJ=$(curl -s -S -m 20 -f "$API/openai-smoke?live=true" || true)
    OLOK=$(python3 - <<'PY'
import sys, json
try:
    j=json.loads(sys.stdin.read()); print(str(bool(j.get("ok", False))).lower())
except Exception:
    print("false")
PY
<<<"$OLJ")
    if [[ "$OLOK" != "true" ]]; then
      say "OpenAI live check FAILED"
      echo "[smoke] Response: $OLJ"
      exit 5
    fi
    say "OpenAI live: OK"
  else
    say "Skipping OpenAI LIVE check (OPENAI_LIVE_SMOKE not true)"
  fi
else
  say "Skipping OpenAI readiness (no key present)"
fi

say "Starting sample job"
JOB_JSON=""
for i in 1 2 3 4 5; do
  JOB_JSON=$(curl -s -S -m 10 -H 'Accept: application/json' -X POST "$API/sample" || true)
  JOB_ID=$(python3 - <<'PY'
import sys, json
try:
    s=sys.stdin.read().strip()
    print(json.loads(s).get("job_id",""))
except Exception:
    print("")
PY
<<<"$JOB_JSON")
  if [[ -n "$JOB_ID" ]]; then
    break
  fi
  say "Retrying /sample ($i) ..."
  sleep 2
done
if [[ -z "$JOB_ID" ]]; then
  say "Failed to parse job_id from /sample"
  echo "[smoke] Raw response from /sample:" >&2
  echo "$JOB_JSON" >&2
  # dump backend logs for debugging
  docker compose logs backend | tail -n 200 || true
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

