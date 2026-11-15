"""GuardAI API.  Run from the project root:

    uvicorn app.main:app --reload

then open http://127.0.0.1:8000/docs to try it in the browser.
"""
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.pipeline import GuardPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = GuardPipeline.load()  # load the model once at startup, not per request
    yield


app = FastAPI(title="GuardAI", version="0.1.0", lifespan=lifespan)


class CheckRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000, description="The prompt to check")


class CheckResponse(BaseModel):
    action: str
    layer: Optional[str]
    ml_score: Optional[float]
    rules: List[str]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "ml_model_loaded": app.state.pipeline.ml is not None}


@app.post("/check", response_model=CheckResponse)
def check(req: CheckRequest):
    return asdict(app.state.pipeline.check(req.text))