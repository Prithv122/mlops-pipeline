"""Health, model-info and prediction endpoints.

Handlers are plain ``def`` on purpose: scoring and the SQLite write are blocking, so
FastAPI runs them on its thread pool instead of stalling the event loop.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, create_model

from mlops_pipeline.model.costs import CostModel, decide
from mlops_pipeline.model.data import FEATURE_COLS

router = APIRouter()

_feature_fields: dict[str, Any] = {
    col: (float | None, Field(allow_inf_nan=False, description="Feature value, or null"))
    for col in FEATURE_COLS
}

#: The 23 model features (required keys, null allowed: a missing value is something the
#: monitor should see as a null, not a field a client can silently forget) plus exposure.
PredictRequest = create_model(
    "PredictRequest",
    **_feature_fields,
    exposure=(float, Field(ge=0, allow_inf_nan=False, description="Money at risk, NT$")),
)


class PredictResponse(BaseModel):
    p_default: float = Field(ge=0, le=1)
    flag: bool
    model_version: str


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/model")
def model_info(request: Request) -> dict[str, Any]:
    loaded = request.app.state.model
    settings = request.app.state.settings
    return {
        "name": settings.model_name,
        "alias": settings.model_alias,
        "version": loaded.version,
        "run_id": loaded.run_id,
        "metrics": loaded.metrics,
    }


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:  # type: ignore[valid-type]
    loaded = request.app.state.model
    payload = body.model_dump()
    exposure = float(payload.pop("exposure"))
    features: dict[str, float | None] = {c: payload[c] for c in FEATURE_COLS}

    frame = pd.DataFrame([features], columns=FEATURE_COLS, dtype="object")
    p = float(loaded.model.predict(frame)[0])
    flag = bool(decide(np.array([p]), np.array([exposure]), CostModel())[0])

    request.app.state.predlog.append(
        ts=datetime.now(UTC),
        model_version=loaded.version,
        features=features,
        exposure=exposure,
        p_default=p,
        flag=flag,
    )
    return PredictResponse(p_default=p, flag=flag, model_version=loaded.version)
