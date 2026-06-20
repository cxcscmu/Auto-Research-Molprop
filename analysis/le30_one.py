"""Single-endpoint fit-then-test worker for parallel ≤30 sweeps. Writes a per-task
file analysis/tdc_test_<label>__<task>.json so many can run concurrently with no
race; a merge step folds them back into analysis/tdc_test_<label>.json.

  argv: <label> <task> <config_dir> [data_external_dir]
Skips if the per-task file already exists. Run with .venv_drug.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import test_eval as te  # noqa: E402

label, task, cfg = sys.argv[1], sys.argv[2], sys.argv[3]
ext = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
out = te.ROOT / f"analysis/tdc_test_{label}__{task}.json"
if out.exists() and json.loads(out.read_text()).get("test") is not None:
    print(f"SKIP {label}/{task}", flush=True)
    sys.exit(0)
venv = str(te.ROOT / ".venv_drug/bin/python")
try:
    r = te.fit_then_test(cfg, "tdc", task, venv, data_external_dir=ext)
    out.write_text(json.dumps({"metric": r["metric"], "val": r["val"], "test": r["test"]}))
    print(f"DONE {label}/{task} val={r['val']} test={r['test']}", flush=True)
except Exception as e:
    print(f"FAIL {label}/{task}: {str(e)[:200]}", flush=True)
    sys.exit(1)
