"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from causalift import __version__
from causalift.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="causalift",
        description="Causal uplift scoring and targeting-policy simulation API",
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()
