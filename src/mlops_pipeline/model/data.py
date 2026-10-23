# Vendored from github.com/Prithv122/fraud-calibrated @ ebf6770, src/fraud_calibrated/data.py
# (MIT, same author). Imports rewritten to this package; logic unchanged.
"""Fetch, clean and split the UCI 'default of credit card clients' dataset.

The source is a legacy BIFF ``.xls`` inside a zip, which ``xlrd`` parses at roughly
6 s per read. Everything downstream needs the frame many times over, so the cleaned
frame is cached as gzipped CSV (~1 MB) and the Excel path runs exactly once per
machine. Parquet would be tidier but pulls in pyarrow for 30k rows; not worth it.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

RAW_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
EXCEL_MEMBER = "default of credit card clients.xls"

#: sha256 of the zip as downloaded on 2026-09-07. UCI regenerates archives
#: occasionally, so a mismatch warns rather than raises -- a hard failure here
#: would break a clean clone for a reason that has nothing to do with this code.
RAW_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"

TARGET = "default"

#: Repayment status columns. The source names the first one PAY_0; every other
#: month is PAY_2..PAY_6, so PAY_1 is missing and PAY_0 is really PAY_1.
PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC_COLS = ["LIMIT_BAL", "AGE", *PAY_COLS, *BILL_COLS, *PAY_AMT_COLS]
FEATURE_COLS = [*NUMERIC_COLS, *CATEGORICAL_COLS]


class DatasetError(RuntimeError):
    """Raised when the archive cannot be fetched or does not look like the dataset."""


def download(dest_dir: str | Path, *, force: bool = False, timeout: float = 60.0) -> Path:
    """Download the source zip into ``dest_dir`` and return its path.

    Skips the network entirely when the file is already present unless ``force``.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "uci350.zip"
    if path.exists() and not force:
        return path
    try:
        with urlopen(RAW_URL, timeout=timeout) as response:
            payload = response.read()
    except OSError as exc:  # pragma: no cover - network failure path
        raise DatasetError(f"could not download {RAW_URL}: {exc}") from exc
    path.write_bytes(payload)
    return path


def checksum_matches(path: str | Path) -> bool:
    """True when the archive hashes to the version this project was built against."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest == RAW_SHA256


def read_archive(path: str | Path) -> pd.DataFrame:
    """Parse the ``.xls`` member of the archive into a raw, uncleaned frame."""
    try:
        with zipfile.ZipFile(path) as archive:
            payload = archive.read(EXCEL_MEMBER)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"{path} is not the expected UCI archive: {exc}") from exc
    # Row 0 holds a merged banner ('X1', 'X2', ...); the real header is row 1.
    return pd.read_excel(io.BytesIO(payload), header=1)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented fixes to the raw frame.

    Four deliberate decisions, all of them visible in the data rather than in the
    UCI description:

    * ``PAY_0`` is renamed ``PAY_1`` -- the source skips PAY_1 and the off-by-one
      name has misled enough published notebooks to be worth fixing at the door.
    * ``EDUCATION`` takes undocumented values 0, 5 and 6; all three are folded into
      category 4 ("other"), which is what the codebook implies they are.
    * ``MARRIAGE`` value 0 is undocumented and folded into 3 ("other").
    * ``ID`` is dropped. It is a row number, and a monotone row number is exactly
      the kind of column a gradient booster will happily leak on.
    """
    df = raw.rename(columns={"PAY_0": "PAY_1", "default payment next month": TARGET}).copy()
    missing = {TARGET, "PAY_1", "LIMIT_BAL"} - set(df.columns)
    if missing:
        raise DatasetError(f"unexpected schema, missing columns: {sorted(missing)}")
    df = df.drop(columns=["ID"], errors="ignore")
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    df[TARGET] = df[TARGET].astype(np.int8)
    return df.reset_index(drop=True)


def load(
    cache_dir: str | Path = "data",
    *,
    force_download: bool = False,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Return the cleaned frame, building the CSV cache on first call."""
    cache_dir = Path(cache_dir)
    cache = cache_dir / "interim" / "credit_default.csv.gz"
    if cache.exists() and not (force_download or force_rebuild):
        cached = pd.read_csv(cache)
        # CSV round-tripping loses dtype information; TARGET must stay int8 so a
        # cache hit and a freshly-cleaned frame are byte-for-byte equivalent.
        cached[TARGET] = cached[TARGET].astype(np.int8)
        return cached
    archive = download(cache_dir / "raw", force=force_download)
    df = clean(read_archive(archive))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, compression="gzip")
    return df


def exposure(df: pd.DataFrame) -> np.ndarray:
    """Money at risk per customer, in NT$.

    The balance that would go unpaid if the customer defaulted next month: the most
    recent statement minus what they just paid, floored at zero (a credit balance is
    not an exposure) and capped at the credit limit (the bank cannot lose more than
    it extended). This is the quantity the cost model in :mod:`mlops_pipeline.model.costs`
    multiplies by loss-given-default.
    """
    unpaid = df["BILL_AMT1"].to_numpy(dtype=float) - df["PAY_AMT1"].to_numpy(dtype=float)
    limit = df["LIMIT_BAL"].to_numpy(dtype=float)
    return np.clip(unpaid, 0.0, limit)


@dataclass(frozen=True)
class Split:
    """A three-way split, kept as frames so cost columns survive alongside features."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "valid": len(self.valid), "test": len(self.test)}


def split(
    df: pd.DataFrame,
    *,
    seed: int = 0,
    valid_frac: float = 0.2,
    test_frac: float = 0.2,
) -> Split:
    """Stratified 60/20/20 split.

    Three ways, not two, because the threshold and the calibrator are both fitted
    on held-out scores. Reusing the test set for either would tune the decision rule
    on the data it is later reported against -- a subtler version of the same leak
    as fitting the scaler before splitting.
    """
    if not 0 < valid_frac + test_frac < 1:
        raise ValueError("valid_frac + test_frac must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "valid": [], "test": []}
    for _, group in df.groupby(TARGET, sort=True):
        order = rng.permutation(len(group))
        shuffled = group.iloc[order]
        n_valid = round(valid_frac * len(group))
        n_test = round(test_frac * len(group))
        parts["valid"].append(shuffled.iloc[:n_valid])
        parts["test"].append(shuffled.iloc[n_valid : n_valid + n_test])
        parts["train"].append(shuffled.iloc[n_valid + n_test :])
    out = {}
    for name, frames in parts.items():
        merged = pd.concat(frames).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        out[name] = merged
    return Split(**out)


def stratum_split(df: pd.DataFrame, column: str, in_values: list[int]) -> tuple[pd.DataFrame, ...]:
    """Split on a real subpopulation, for the covariate-shift experiment.

    Returns ``(in_domain, out_of_domain)``. Unlike a random split this produces a
    genuine distribution shift that exists in the data, so the drift numbers in the
    README are measured rather than injected.
    """
    if column not in df.columns:
        raise ValueError(f"{column!r} is not a column of the frame")
    mask = df[column].isin(in_values).to_numpy()
    return df.loc[mask].reset_index(drop=True), df.loc[~mask].reset_index(drop=True)
