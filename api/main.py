"""FastAPI app.  Run: uvicorn api.main:app --reload   (Swagger UI at /docs)"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import config
from api.routes.core import router
from api.routes.realworld import router as realworld_router
from ingestion import scheduler
from ingestion.service import IngestionService
from nlp.pipeline import Pipeline


def create_app(pipeline: Pipeline | None = None, service: IngestionService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config.AUTOPOLL_IN_API:  # optional; the lease lock makes duplicate pollers harmless
            scheduler.start_background(app.state.service)
        yield
        scheduler.stop_background()

    app = FastAPI(title="ResQText API", version="2.0.0", lifespan=lifespan,
                  description=f"Disaster message intelligence with real-world sources. **{config.DISCLAIMER}**")
    app.state.pipeline = pipeline or Pipeline()
    app.state.service = service or IngestionService(app.state.pipeline.db, app.state.pipeline)
    app.include_router(router)
    app.include_router(realworld_router)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_, exc: RequestValidationError):
        errs = [{"field": ".".join(str(x) for x in e["loc"][1:]), "problem": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "Invalid request", "errors": errs})

    return app


app = create_app()
