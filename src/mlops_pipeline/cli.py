"""Command line: prepare | train | serve | monitor | inject | eval-responder | retrain.

Heavy imports (MLflow, LightGBM, FastAPI, Evidently) happen inside each command so that
``mlops-pipeline --help`` stays fast and a command only pays for what it uses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

DEFAULT_DATA = "data/credit_default.csv.gz"
DEFAULT_TRACKING = "sqlite:///mlstore/mlflow.db"

#: What the one-time evaluation is allowed to run on (PREREGISTRATION.md, Amendment 2).
PROTOCOL_WINDOW_SIZE = 1000
#: SHA-256 of the vendored data cache, ``data/credit_default.csv.gz``.
PROTOCOL_DATA_SHA256 = "c4fdd4c967cba262fd21be076cddece7ca18fd052eb19fb755b4938919c143f9"


def _read(path: str):
    import pandas as pd

    from mlops_pipeline.model.data import TARGET

    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype("int8")
    return df


def _parse_time(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path | None:
    """The folder holding ``pyproject.toml`` above this file, or None for an installed copy."""
    for folder in Path(__file__).resolve().parents:
        if (folder / "pyproject.toml").is_file():
            return folder
    return None


def _git_state() -> tuple[str | None, bool]:
    """Commit and cleanliness of the repository this code lives in; (None, False) outside git.

    Git runs with ``-C`` on the repository root found from this file, so the answer does not
    depend on the folder the command was started from.
    """
    root = _repo_root()
    if root is None:
        return None, False
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return sha, not dirty


def _library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "mlops-pipeline"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "unknown"
    return out


def cmd_prepare(args: argparse.Namespace) -> int:
    from mlops_pipeline.model.data import split

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = split(_read(args.data), seed=args.seed)
    for name, frame in (("train", parts.train), ("valid", parts.valid), ("test", parts.test)):
        frame.to_csv(
            out / f"{name}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
        )
    print(json.dumps(parts.sizes))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from mlops_pipeline.registry import CHAMPION, log_and_register
    from mlops_pipeline.train import train

    result = train(_read(args.data), seed=args.seed)
    version = log_and_register(
        result,
        tracking_uri=args.tracking_uri,
        data_path=args.data,
        alias=None if args.no_promote else CHAMPION,
    )
    summary = {"version": version, **result.metrics}
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from mlops_pipeline.registry import retrain

    decision = retrain(
        _read(args.data), tracking_uri=args.tracking_uri, seed=args.seed, data_path=args.data
    )
    print(json.dumps(asdict(decision)))
    return 0


def _load_null_model(path: str | None, digest: str, window_size: int):
    """A saved null model, or None when it is missing or was not fitted for this setup."""
    from mlops_pipeline.incident.ranker import NullModel

    if not path or not Path(path).exists():
        return None
    try:
        null = NullModel.from_json(path)
    except (ValueError, KeyError):
        return None
    if null.reference_sha256 != digest or null.window_size != window_size:
        return None
    return null


def _responder(settings, null_model_path: str | None, *, window_size: int = 1000):
    """Reference frame and profile plus the no-fault calibration (loaded from JSON when given).

    The reference is the train split of ``settings.reference_path``, the same rows the
    evaluation compares windows with, so the API and the scored responder are one thing. A
    saved calibration is trusted only if it was fitted on this reference file at this window
    size; otherwise it is refitted and overwritten, because a stale one scores the wrong data.
    The responder exposes ``min_rows``: the scores are only calibrated for windows that long.
    """
    import dataclasses

    from mlops_pipeline.evaluate import fit_responder
    from mlops_pipeline.incident.analyser import analyse
    from mlops_pipeline.model.data import split
    from mlops_pipeline.monitor.stats import profile

    df = _read(settings.reference_path)
    reference = split(df, seed=0).train
    digest = _file_sha256(settings.reference_path)
    null = _load_null_model(null_model_path, digest, window_size)
    if null is not None:
        ref = profile(reference)
    else:
        ref, fitted = fit_responder(df, window_size=window_size)
        null = dataclasses.replace(fitted, reference_sha256=digest)
        if null_model_path:
            null.to_json(null_model_path)

    class _Responder:
        min_rows = null.window_size

        def respond(self, window):
            return analyse(ref, null, window)

    return reference, ref, null, _Responder()


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from mlops_pipeline.serve.app import create_app
    from mlops_pipeline.serve.settings import get_settings

    settings = get_settings()
    *_, responder = _responder(settings, args.null_model)
    uvicorn.run(create_app(settings, responder=responder), host=args.host, port=args.port)
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    from mlops_pipeline.monitor.run import monitor_window, post_alert
    from mlops_pipeline.predlog import PredictionLog
    from mlops_pipeline.serve.settings import get_settings

    settings = get_settings()
    reference, ref, null, _ = _responder(settings, args.null_model)
    alerts: list[dict] = []

    def record(alert) -> None:
        alerts.append(alert.model_dump(mode="json"))
        if args.webhook:
            post_alert(args.webhook)(alert)

    result = monitor_window(
        PredictionLog(settings.predlog_path),
        ref,
        null,
        start=_parse_time(args.start),
        end=_parse_time(args.end),
        alert=record,
        report_dir=Path(args.report_dir) if args.report_dir else None,
        reference_frame=reference,
    )
    ranking = result.ranking
    top = ranking.top if ranking else None
    print(
        json.dumps(
            {
                "status": result.status,
                "rows": result.rows,
                "min_rows": result.min_rows,
                "incident": bool(ranking and ranking.incident),
                "top": None if top is None else {"cause": top.cause, "column": top.column},
                "alerts": alerts,
            }
        )
    )
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    from mlops_pipeline.contracts import CONTROL
    from mlops_pipeline.incident.faults import dev_grid, draw_window
    from mlops_pipeline.model.data import split

    if args.seed < 10_000:
        print("inject uses development seeds only (>= 10000); evaluation seeds stay sealed")
        return 2
    # Build the scenario through the grid itself so the affected columns follow its rules.
    n = args.seed - 10_000 + 1
    matches = [
        s
        for s in dev_grid(n_seeds=n, n_controls=n)
        if s.fault == args.fault
        and s.seed == args.seed
        and (args.fault == CONTROL or s.severity == args.severity)
    ]
    if not matches:
        print(f"no development scenario for fault={args.fault} severity={args.severity}")
        return 2
    scenario = matches[0]
    parts = split(_read(args.data), seed=0)
    window = draw_window(parts.test, scenario, parts.train, size=args.window_size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    window.to_csv(args.out, index=False)
    print(json.dumps({"scenario": scenario.scenario_id, "columns": list(scenario.columns)}))
    return 0


def _refuse_full_run(args: argparse.Namespace, clean: bool) -> str | None:
    """Why the one-time evaluation may not start, or None when every guard passes."""
    if args.window_size != PROTOCOL_WINDOW_SIZE:
        return (
            f"the full evaluation uses windows of {PROTOCOL_WINDOW_SIZE} rows, "
            f"not {args.window_size}"
        )
    if not Path(args.data).is_file():
        return f"data file not found: {args.data}"
    if _file_sha256(args.data) != PROTOCOL_DATA_SHA256:
        return "the data file is not the vendored cache the protocol was frozen on (hash differs)"
    if not clean and not args.allow_dirty:
        return "the full evaluation runs once, from a clean committed tree; commit first"
    if (Path(args.out) / "records.csv").exists():
        return f"{args.out} already holds records.csv; the full evaluation runs once"
    return None


def cmd_eval_responder(args: argparse.Namespace) -> int:
    from mlops_pipeline.evaluate import run_evaluation, summarise, write_report
    from mlops_pipeline.incident.faults import dev_grid, eval_grid

    sha, clean = _git_state()
    if args.full:
        reason = _refuse_full_run(args, clean)
        if reason:
            print(reason)
            return 2
        scenarios = eval_grid()
    else:
        scenarios = dev_grid(n_seeds=args.dev_seeds, n_controls=args.dev_controls)
    started = datetime.now(UTC)
    records = run_evaluation(_read(args.data), scenarios, window_size=args.window_size)
    summary = summarise(records)
    summary["provenance"] = {
        "mode": "full" if args.full else "smoke",
        "data": args.data,
        "data_sha256": _file_sha256(args.data),
        "window_size": args.window_size,
        "commit": sha,
        "tree_clean": clean,
        "allow_dirty": bool(args.allow_dirty),
        "python": platform.python_version(),
        "versions": _library_versions(),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_report(records, summary, Path(args.out))
    print(f"wrote {len(records)} scenario records to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlops-pipeline", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="write the train/valid/test split")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default="data/processed")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="train, log to MLflow, register (alias champion)")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-promote", action="store_true", help="register without the alias")
    p.add_argument("--metrics-out", help="also write the metrics as JSON here")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("retrain", help="train a challenger; promote only by the rule")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_retrain)

    p = sub.add_parser("serve", help="run the API (settings from MLOPS_* variables)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--null-model", help="no-fault calibration JSON; fitted and saved if missing")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("monitor", help="check one window of the prediction log")
    p.add_argument("--start", required=True, help="ISO time, inclusive")
    p.add_argument("--end", required=True, help="ISO time, exclusive")
    p.add_argument("--webhook", help="POST an alert here when an incident is found")
    p.add_argument("--report-dir", help="also write the Evidently drift report here")
    p.add_argument("--null-model")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("inject", help="write one faulted window (development seeds only)")
    p.add_argument("--fault", required=True)
    p.add_argument("--severity", type=int, default=2)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--window-size", type=int, default=1000)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_inject)

    p = sub.add_parser("eval-responder", help="score the ranker against the PSI baseline")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="development seeds only")
    mode.add_argument("--full", action="store_true", help="the pre-registered grid, run once")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", required=True)
    p.add_argument("--window-size", type=int, default=1000)
    p.add_argument("--dev-seeds", type=int, default=2)
    p.add_argument("--dev-controls", type=int, default=10)
    p.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_eval_responder)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
