"""Drug Discovery experiment entry point (seed_file / baseline_filename).

Contract with run_trial_drug.py (harness-controlled, do NOT change signatures):

  Fit mode:
    python experiment.py --task <name> --mode fit \
        --train <train_csv_with_Y> --val-x <val_smiles_csv_no_Y> \
        --model-dir <dir>

  Predict mode:
    python experiment.py --task <name> --mode predict \
        --input <smiles_csv_no_Y> --output <pred_csv> --model-dir <dir>

  --val-x: validation SMILES only (Drug / Drug_ID columns, NO Y column).
    This is intentional: the harness holds val labels and evaluates
    predictions externally. The agent pipeline NEVER receives val Y.

SMOKE_TEST=1: exit immediately with dummy output.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.pipeline import DrugPipeline


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task",      required=True)
    p.add_argument("--mode",      required=True, choices=["fit", "predict"])
    p.add_argument("--train",     help="Train CSV with Y (fit mode)")
    p.add_argument("--val-x",     dest="val_x",
                   help="Val SMILES CSV — NO Y column (fit mode, optional)")
    p.add_argument("--input",     help="SMILES CSV, no Y (predict mode)")
    p.add_argument("--output",    help="Output predictions CSV (predict mode)")
    p.add_argument("--model-dir", default=".", dest="model_dir")
    args = p.parse_args()

    smoke = os.environ.get("SMOKE_TEST", "0") == "1"
    pipeline  = DrugPipeline()
    model_path = Path(args.model_dir) / f"model_{args.task}.pkl"

    if args.mode == "fit":
        if smoke:
            model_path.write_bytes(b"smoke")
            print(f"SMOKE fit done for {args.task}")
            return 0
        import pandas as pd
        train_df = pd.read_csv(args.train)
        # val_x has NO Y column — agent cannot memorise val labels
        val_x_df = pd.read_csv(args.val_x) if args.val_x else None
        pipeline.fit(train_df, val_x_df, args.task)
        pipeline.save(str(model_path))
        return 0

    if args.mode == "predict":
        if smoke:
            import pandas as pd
            test_df = pd.read_csv(args.input)
            pd.DataFrame({
                "Drug_ID": test_df.get("Drug_ID", range(len(test_df))),
                "Y":       [0.5] * len(test_df),
            }).to_csv(args.output, index=False)
            print(f"SMOKE predict done for {args.task}")
            return 0
        import pandas as pd
        test_df = pd.read_csv(args.input)
        pipeline.load(str(model_path))
        preds = pipeline.predict(test_df, args.task)
        pd.DataFrame({
            "Drug_ID": test_df.get("Drug_ID", range(len(test_df))),
            "Y":       preds,
        }).to_csv(args.output, index=False)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
