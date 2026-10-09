"""
FastAPI service for churn prediction.

Run from the project root:

    uvicorn app.main:app --reload

then open http://127.0.0.1:8000/docs for interactive documentation.

Endpoints:
    GET  /health      is the service up (used by load balancers)
    GET  /model-info  which model is serving and how it scored in training
    POST /predict     score one customer

The model is loaded once at startup, not on every request. Requests are
validated by the Pydantic models in schemas.py; the prediction itself is
done by ChurnPredictor, the same class any other client (such as the UI)
uses, so every entry point gives the same answer.

The model bundle is read from the MODEL_DIR environment variable, or from
the project's model/ folder if it is not set.
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from app.schemas import (
    CustomerData,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.serving.inference import ChurnPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model before the first request is served."""
    app.state.predictor = ChurnPredictor()
    yield


app = FastAPI(
    title="Telco Customer Churn API",
    description=(
        "Predicts whether a telecom customer is likely to churn. The model "
        "is tuned to catch most churners (high recall) while keeping the "
        "flagged list as accurate as possible."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_predictor(request: Request) -> ChurnPredictor:
    """Give an endpoint the predictor that was loaded at startup."""
    return request.app.state.predictor


Predictor = Annotated[ChurnPredictor, Depends(get_predictor)]


@app.get("/health", response_model=HealthResponse)
def health():
    """The service starts after the model has loaded, so 'ok' means ready."""
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(predictor: Predictor):
    """Details of the model that is currently serving."""
    return predictor.info


# A plain `def` (not `async def`): scoring is CPU work, so FastAPI runs it in
# a worker thread instead of blocking the event loop.
@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData, predictor: Predictor):
    """Score one customer. Invalid input is rejected with a 422 error."""
    return predictor.predict(customer.model_dump())
