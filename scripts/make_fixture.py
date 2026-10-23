"""Build the 3,000-row test fixture from the vendored data cache.

Stratified on the target with a fixed seed, so the fixture is reproducible and keeps the
full data's default rate. Run once; the output is committed and tests read only it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlops_pipeline.model.data import TARGET

SEED = 20260923
N_ROWS = 3000
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "credit_default.csv.gz")
    frac = N_ROWS / len(df)
    sample = (
        df.groupby(TARGET, group_keys=False)
        .sample(frac=frac, random_state=SEED)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )
    out = ROOT / "tests" / "fixtures" / "credit_sample.csv.gz"
    sample.to_csv(out, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"wrote {len(sample)} rows, default rate {sample[TARGET].mean():.4f} -> {out}")


if __name__ == "__main__":
    main()
