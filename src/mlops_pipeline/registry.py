"""MLflow logging, registration, aliases and the retrain decision.

The backend is whatever ``tracking_uri`` names; the project runs on a local SQLite file, so
no tracking server is needed. The model is logged as a ``pyfunc`` whose only artifact is the
pickled :class:`CalibratedModel`. Pickling (rather than exporting the booster to text and
rebuilding it) is what makes a reload bit-for-bit identical, which the tests assert.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

# Set before mlflow is imported: it prints an unrelated hint line on import otherwise. The
# second switch stops it copying the whole repo's lock file into every model version, since
# the pinned requirements are already recorded explicitly below.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_UV_AUTO_DETECT", "false")

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mlops_pipeline.model.data import split
from mlops_pipeline.train import CalibratedModel, TrainResult, score_model, train

MODEL_NAME = "credit-default"
CHAMPION = "champion"

#: Challenger may lose at most this much AUC, and gain at most this much Brier, and still
#: be promoted. Small tolerances so noise between two seeds does not block a fresh model.
AUC_TOLERANCE = 0.005
BRIER_TOLERANCE = 0.002

_ARTIFACT_KEY = "calibrated"
_PICKLE_NAME = "calibrated_model.pkl"


@dataclass(frozen=True)
class LoadedModel:
    model: CalibratedModel
    version: str
    run_id: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class RetrainDecision:
    promoted: bool
    challenger_version: str
    champion_version: str | None
    reason: str


class _CalibratedPyfunc(mlflow.pyfunc.PythonModel):
    """Wrapper so the registry holds a standard model flavour around the pickle."""

    def load_context(self, context) -> None:
        with open(context.artifacts[_ARTIFACT_KEY], "rb") as fh:
            self.calibrated: CalibratedModel = pickle.load(fh)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return self.calibrated.predict(model_input)


def _store_dir(tracking_uri: str) -> Path | None:
    """Folder holding the SQLite file, or None for any other kind of backend."""
    if not tracking_uri.startswith("sqlite:"):
        return None
    raw = tracking_uri[len("sqlite:") :]
    if raw.startswith("///"):
        raw = raw[3:]  # sqlite:///rel.db -> rel.db ; sqlite:////abs.db -> /abs.db
    elif raw.startswith("//"):
        raw = urlparse(tracking_uri).path
    path = Path(unquote(raw))
    # On Windows 'sqlite:///C:/x/m.db' leaves 'C:/x/m.db'; 'sqlite:////C:/..' leaves '/C:/..'.
    if len(raw) > 2 and raw[0] == "/" and raw[2] == ":":
        path = Path(raw[1:])
    return path.resolve().parent


def _client(tracking_uri: str) -> MlflowClient:
    store = _store_dir(tracking_uri)
    if store is not None:
        # SQLite creates the file but not its folder; a fresh clone has no mlstore/ yet.
        store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def _experiment_id(client: MlflowClient, tracking_uri: str, name: str) -> str:
    """Experiment whose artifacts sit beside the SQLite file, not in the working directory.

    MLflow's default artifact root is ``./mlruns`` relative to wherever the process started,
    which scatters files across the repo and breaks as soon as the server is launched from
    another folder. An explicit location fixes both.
    """
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    store = _store_dir(tracking_uri)
    location = (store / "mlartifacts").as_uri() if store is not None else None
    return client.create_experiment(name, artifact_location=location)


def _sha256(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pip_requirements() -> list[str]:
    names = ["lightgbm", "scikit-learn", "numpy", "pandas", "scipy", "mlflow"]
    return [f"{n}=={metadata.version(n)}" for n in names]


def log_and_register(
    result: TrainResult,
    *,
    tracking_uri: str,
    model_name: str = MODEL_NAME,
    data_path: str,
    alias: str | None = CHAMPION,
) -> str:
    """Log one training run, register it as a new model version and return the version.

    ``alias=None`` registers without serving it: that is how a challenger waits for the
    promotion decision.
    """
    client = _client(tracking_uri)
    experiment_id = _experiment_id(client, tracking_uri, model_name)

    booster_params = {
        k: v
        for k, v in result.model.booster.get_params().items()
        if isinstance(v, str | int | float | bool) or v is None
    }
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(experiment_id=experiment_id):
        pkl = Path(tmp) / _PICKLE_NAME
        pkl.write_bytes(pickle.dumps(result.model))
        mlflow.log_params({f"lgbm_{k}": v for k, v in booster_params.items()})
        mlflow.log_params(
            {
                "n_train": result.n_train,
                "n_valid": result.n_valid,
                "n_test": result.n_test,
                "data_path": data_path,
            }
        )
        digest = _sha256(data_path)
        if digest:
            mlflow.set_tag("data_sha256", digest)
        mlflow.log_metrics(result.metrics)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=_CalibratedPyfunc(),
            artifacts={_ARTIFACT_KEY: str(pkl)},
            registered_model_name=model_name,
            pip_requirements=_pip_requirements(),
        )
    version = str(info.registered_model_version)
    if alias:
        client.set_registered_model_alias(model_name, alias, version)
    return version


def load_model(
    *, tracking_uri: str, model_name: str = MODEL_NAME, alias: str = CHAMPION
) -> LoadedModel:
    """Load the model version an alias points at, with its logged metrics."""
    client = _client(tracking_uri)
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except MlflowException as exc:
        raise LookupError(
            f"no model {model_name!r} with alias {alias!r} in {tracking_uri}"
        ) from exc
    loaded = mlflow.pyfunc.load_model(f"models:/{model_name}/{mv.version}")
    calibrated = loaded.unwrap_python_model().calibrated
    run_id = mv.run_id or ""
    metrics = dict(client.get_run(run_id).data.metrics) if run_id else {}
    return LoadedModel(model=calibrated, version=str(mv.version), run_id=run_id, metrics=metrics)


def _score_champion(champion: LoadedModel, test: pd.DataFrame) -> dict[str, float]:
    """Champion metrics on the challenger's test split.

    Its logged metrics come from whatever split it was trained with, so they are not
    comparable; scoring both on one frame is. If the champion saw some of these rows in
    training the comparison favours it, which only makes promotion harder.
    """
    return score_model(champion.model, test)


def retrain(
    df: pd.DataFrame,
    *,
    tracking_uri: str,
    model_name: str = MODEL_NAME,
    seed: int = 0,
    data_path: str = "",
) -> RetrainDecision:
    """Train a challenger, register it without an alias, promote it only by the rule.

    Promote iff challenger AUC >= champion AUC - 0.005 and challenger Brier <= champion
    Brier + 0.002, both measured on the same test split. With no champion yet the first
    model is promoted unconditionally.
    """
    result = train(df, seed=seed)
    version = log_and_register(
        result, tracking_uri=tracking_uri, model_name=model_name, data_path=data_path, alias=None
    )
    client = _client(tracking_uri)
    try:
        champion = load_model(tracking_uri=tracking_uri, model_name=model_name)
    except LookupError:
        client.set_registered_model_alias(model_name, CHAMPION, version)
        return RetrainDecision(True, version, None, "no champion registered; first model promoted")

    test = split(df, seed=seed).test
    champ = _score_champion(champion, test)
    chal = {k: result.metrics[k] for k in ("test_auc", "test_brier")}
    auc_ok = chal["test_auc"] >= champ["test_auc"] - AUC_TOLERANCE
    brier_ok = chal["test_brier"] <= champ["test_brier"] + BRIER_TOLERANCE
    numbers = (
        f"AUC {chal['test_auc']:.4f} vs {champ['test_auc']:.4f}, "
        f"Brier {chal['test_brier']:.4f} vs {champ['test_brier']:.4f}"
    )
    if auc_ok and brier_ok:
        client.set_registered_model_alias(model_name, CHAMPION, version)
        return RetrainDecision(True, version, champion.version, f"promoted: {numbers}")
    failed = [n for n, ok in (("AUC", auc_ok), ("Brier", brier_ok)) if not ok]
    return RetrainDecision(
        False,
        version,
        champion.version,
        f"kept champion ({' and '.join(failed)} outside the margin): {numbers}",
    )
