from __future__ import annotations
import argparse
import csv
import yaml
from pathlib import Path

HTML_TMPL = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Bench Report</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial; padding: 16px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; font-size: 14px; }
    tr:nth-child(even){background-color:#f8fafc}
    tr:hover{background-color:#eef2ff}
    .pass { color: #0a7; font-weight: 600; }
    .fail { color: #c00; font-weight: 600; }
  </style>
</head>
<body>
<h1>Benchmark Report</h1>
<p>Thresholds: primary metric >= {threshold} (per-dataset overrides applied)</p>
<table>
<thead><tr><th>Dataset</th><th>Status</th><th>Stage</th><th>Primary</th><th>Durations</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
"""

TR = "<tr><td>{dataset}</td><td class=\"{cls}\">{status}</td><td>{stage}</td><td>{primary}</td><td>{dur}</td></tr>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="bench.html")
    ap.add_argument("--thresholds", default=str(Path(__file__).resolve().parent / "thresholds.yaml"))
    ap.add_argument("--default-threshold", type=float, default=0.6)
    args = ap.parse_args()

    overrides = {}
    try:
        with open(args.thresholds, "r") as tf:
            cfg = yaml.safe_load(tf) or {}
            overrides = cfg.get("thresholds", {}) or {}
    except Exception:
        overrides = {}

    rows = []
    with open(args.csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ds = row.get("dataset")
            primary = row.get("metric_primary") or ""
            status = row.get("status") or ""
            th = float(overrides.get(ds, args.default_threshold))
            cls = "pass" if (status == "COMPLETED" and (primary == "" or float(primary) >= th)) else "fail"
            dur = row.get("durations_ms") or ""
            rows.append(TR.format(dataset=ds, cls=cls, status=status, stage=row.get("stage"), primary=primary, dur=dur))

    html = HTML_TMPL.format(rows="\n".join(rows), threshold=args.default_threshold)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()

