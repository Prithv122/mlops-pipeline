from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def test_dvc_stages_and_commands():
    stages = load("dvc.yaml")["stages"]
    assert list(stages) == ["prepare", "train", "eval_responder_smoke"]
    assert stages["prepare"]["cmd"] == (
        "mlops-pipeline prepare --data data/credit_default.csv.gz --out data/processed"
    )
    assert stages["train"]["cmd"] == (
        "mlops-pipeline train --data data/credit_default.csv.gz "
        "--tracking-uri sqlite:///mlstore/mlflow.db"
    )
    assert stages["eval_responder_smoke"]["cmd"] == (
        "mlops-pipeline eval-responder --smoke --data data/credit_default.csv.gz "
        "--out results/smoke"
    )


def test_dvc_outputs_deps_and_metrics():
    stages = load("dvc.yaml")["stages"]
    assert stages["prepare"]["outs"] == ["data/processed"]
    assert "data/credit_default.csv.gz" in stages["prepare"]["deps"]
    assert "src/mlops_pipeline/train.py" in stages["train"]["deps"]
    assert stages["train"]["outs"] == [{"mlstore": {"cache": False}}]
    assert stages["train"]["metrics"] == [{"metrics/train.json": {"cache": False}}]
    assert stages["eval_responder_smoke"]["outs"] == ["results/smoke"]
    for stage in stages.values():
        for dep in stage["deps"]:
            assert "__pycache__" not in dep


def test_retrain_workflow_is_manual_and_secret_free():
    text = (ROOT / ".github/workflows/retrain.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    triggers = workflow.get("on", workflow.get(True))  # YAML reads a bare `on` as True
    assert list(triggers) == ["workflow_dispatch"]
    assert "${{ secrets" not in text
    steps = workflow["jobs"]["retrain"]["steps"]
    assert any(s.get("uses", "").startswith("actions/upload-artifact@v4") for s in steps)
    assert any("retrain --data data/credit_default.csv.gz" in s.get("run", "") for s in steps)


def test_ci_runs_the_smoke_evaluation_after_tests():
    steps = load(".github/workflows/ci.yml")["jobs"]["test"]["steps"]
    runs = [s.get("run", "") for s in steps]
    test_at = next(i for i, r in enumerate(runs) if "pytest" in r)
    smoke_at = next(i for i, r in enumerate(runs) if "eval-responder --smoke" in r)
    assert smoke_at > test_at


def test_env_example_lists_exactly_the_variables_the_code_reads():
    names = {
        line.split("=", 1)[0]
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert names == {
        "MLOPS_TRACKING_URI",
        "MLOPS_MODEL_NAME",
        "MLOPS_MODEL_ALIAS",
        "MLOPS_PREDLOG_PATH",
        "MLOPS_REFERENCE_PATH",
        "MLOPS_POSTMORTEM_DIR",
        "MLOPS_SLACK_WEBHOOK_URL",
    }
