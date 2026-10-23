"""App factory. Everything the routes need hangs off ``app.state``."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mlops_pipeline.predlog import PredictionLog
from mlops_pipeline.registry import LoadedModel, load_model
from mlops_pipeline.serve.routes import router
from mlops_pipeline.serve.settings import Settings, get_settings


def create_app(
    settings: Settings | None = None,
    *,
    model: LoadedModel | None = None,
    responder: Any | None = None,
    extra_routers: Sequence[APIRouter] = (),
) -> FastAPI:
    """Build the API.

    With ``model=None`` the champion is loaded from the registry at startup and the app
    refuses to start if there is none: serving nothing is better than serving a guess.
    An injected ``model`` is for tests and is available immediately, which matters because
    ASGI test clients do not run the startup hook.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "model", None) is None:
            app.state.model = load_model(
                tracking_uri=settings.tracking_uri,
                model_name=settings.model_name,
                alias=settings.model_alias,
            )
        yield

    app = FastAPI(title="Credit default scoring", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.model = model
    app.state.responder = responder
    app.state.predlog = PredictionLog(settings.predlog_path)

    @app.exception_handler(RequestValidationError)
    async def _reject(_: Request, exc: RequestValidationError) -> JSONResponse:
        # The default handler echoes the offending input back; a NaN or infinity there makes
        # the JSON encoder raise, turning a client error into a 500. Report where and why only.
        detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    app.include_router(router)
    for extra in extra_routers:
        app.include_router(extra)
    return app
