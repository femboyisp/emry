# Migrating to Emry / importing existing logs

Emry's metric log, `metrics.jsonl`, is deliberately a plain, wide JSONL file so
it stays readable by spreadsheets, `jq`, pandas, and other loggers' tooling —
and so you can import history from another logger by writing the same shape.

This guide documents the schema and shows how to convert existing logs into it.

## The `metrics.jsonl` schema

One JSON object per line ("wide" rows — all of a step's metrics together):

```json
{"step":0,"epoch":0,"phase":"TRAIN","values":{"loss":2.31,"lr":0.0003}}
{"step":1,"epoch":0,"phase":"TRAIN","values":{"loss":2.10,"lr":0.0003}}
{"step":200,"epoch":1,"phase":"EVAL","values":{"loss":1.84,"acc":0.62}}
```

| Field    | Type                  | Notes                                                       |
| -------- | --------------------- | ----------------------------------------------------------- |
| `step`   | integer (u64)         | Global training step. **Required.**                         |
| `epoch`  | integer (u32)         | Current epoch.                                              |
| `phase`  | string                | One of `TRAIN`, `EVAL`, `TEST`, `WARMUP`, `CHECKPOINT`.     |
| `values` | object `{name: number}` | Resolved metric name → value for this step. **Required.** |

Notes:

- Values are keyed by **name**, not id — the file is self-describing.
- `phase` is SCREAMING_SNAKE_CASE.
- Different rows may carry different metrics (e.g. `acc` only on `EVAL` rows);
  consumers treat a missing metric as absent for that step.
- Lines are append-only and independent, so a truncated file (e.g. a preempted
  run) is still valid up to its last complete line.

### Minimal vs. full rows

How many fields you need depends on which tools you point at the file:

- **Live observers** — `emry watch` and `emry web` — parse leniently: only
  `step` and `values` are required. `epoch` defaults to `0` and `phase` to
  `TRAIN`. So the smallest watchable row is `{"step":0,"values":{"loss":2.3}}`.
- **Export and compare** — `emry export csv|parquet` and `emry compare` — parse
  strictly and require **all four** fields. A row missing `epoch` or `phase` is
  reported as a corrupt line rather than silently skipped.

**Recommendation:** always write the full four-field row. It costs nothing and
works with every Emry tool.

## Importing from another logger

The recipe is the same regardless of source: for each logged step, emit one
full row. A few common conversions follow.

### From a CSV (`step,loss,lr,…`)

A wide CSV with a header maps almost directly. With Python:

```python
import csv, json

with open("history.csv") as src, open("metrics.jsonl", "w") as dst:
    for row in csv.DictReader(src):
        step = int(row.pop("step"))
        values = {k: float(v) for k, v in row.items() if v != ""}
        dst.write(json.dumps({
            "step": step, "epoch": 0, "phase": "TRAIN", "values": values,
        }) + "\n")
```

### From long-format records (`{step, name, value}`)

Some loggers emit one row per metric. Group them by step into wide rows:

```python
import json
from collections import defaultdict

rows = defaultdict(dict)            # step -> {name: value}
for rec in read_long_records():     # your source iterator
    rows[rec["step"]][rec["name"]] = float(rec["value"])

with open("metrics.jsonl", "w") as dst:
    for step in sorted(rows):
        dst.write(json.dumps({
            "step": step, "epoch": 0, "phase": "TRAIN", "values": rows[step],
        }) + "\n")
```

### From TensorBoard event files

Read scalars with the TensorBoard API and group by step:

```python
import json
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

acc = EventAccumulator("path/to/tb_logdir")
acc.Reload()

rows = defaultdict(dict)
for tag in acc.Tags()["scalars"]:
    for ev in acc.Scalars(tag):
        rows[ev.step][tag] = ev.value

with open("metrics.jsonl", "w") as dst:
    for step in sorted(rows):
        dst.write(json.dumps({
            "step": step, "epoch": 0, "phase": "TRAIN", "values": rows[step],
        }) + "\n")
```

> Map your own phase/epoch information into the `phase` and `epoch` fields if you
> have it; otherwise the `TRAIN` / `0` defaults above are fine.

## Using the imported file

Emry's read-side tools accept either a **run directory** or a `metrics.jsonl`
file directly:

```bash
# Live view (lenient parser):
emry watch metrics.jsonl
emry web --run-dir metrics.jsonl          # then open http://127.0.0.1:8787

# Export / compare (strict parser — needs full rows):
emry export csv --run-dir metrics.jsonl --output history.csv
emry compare old_run/ metrics.jsonl
```

To make it a first-class run that `emry runs` lists, place the file in a run
directory named `{project}_{YYYYMMDD_HHMMSS}` under your log dir. `emry runs`
reads `run.meta` (`{"run_id","project","start_time_secs","mode"}`) for the
project name and, if present, `summary.json` for the step count and finished
status; add those JSON files if you want that metadata to show. For charting and
export, the `metrics.jsonl` file alone is enough.

## Exporting back out

The same schema flows outward: `emry export csv` and `emry export parquet`
(behind the `parquet` build feature) turn a run's `metrics.jsonl` into a flat
table with `step,epoch,phase` columns plus one column per metric — so moving
data *out* of Emry into other tooling is symmetric with importing it.
