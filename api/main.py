"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.db import init_db
from api.routes import router
from api.storage import ensure_storage_root


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_storage_root()
    init_db()
    yield


app = FastAPI(title="Object Identification Service", lifespan=lifespan)
app.include_router(router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}
