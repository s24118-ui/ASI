"""API FastAPI dla modelu oceny kredytowej.

Uruchomienie lokalnie (z katalogu głównego projektu):

    uvicorn credit_scoring.serving.api:app --reload --app-dir src

Dokumentacja interaktywna (Swagger UI) będzie dostępna pod:

    http://127.0.0.1:8000/docs

Endpointy:
    GET  /health                 — czy API żyje i czy model jest wczytany
    GET  /model/metrics          — metryki modelu zapisane przez pipeline `modeling`
    POST /predict                — predykcja dla jednego klienta
    POST /predict/batch          — predykcja dla listy klientów
    GET  /predictions/stats      — statystyki logowania predykcji (licznik, ostatnia weryfikacja)
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_scoring.serving import inference
from credit_scoring.serving.prediction_logger import prediction_logger
from credit_scoring.serving.schema import (
    LOAN_TYPES,
    OCCUPATIONS,
    PAYMENT_BEHAVIOURS,
    TARGET_LABELS,
)

# Stan aplikacji (model wczytywany raz, przy starcie) ------------------------

_app_state: dict[str, Any] = {"model": None, "model_version": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _app_state["model"] = inference.load_model()
        # Prosty "identyfikator wersji" modelu — rozmiar + czas modyfikacji pliku.
        stat = inference.MODEL_PATH.stat()
        _app_state["model_version"] = f"{stat.st_size}-{int(stat.st_mtime)}"
    except inference.ModelNotAvailableError as exc:
        # Aplikacja wstaje, ale /health i /predict zgłoszą błąd 503 z opisem.
        _app_state["model"] = None
        _app_state["model_error"] = str(exc)
    yield


app = FastAPI(
    title="Credit Scoring API",
    description="API do predykcji oceny kredytowej (projekt ASI / Kedro).",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_model() -> Any:
    model = _app_state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=_app_state.get(
                "model_error", "Model nie jest wczytany. Sprawdź /health."
            ),
        )
    return model


# Schematy żądań / odpowiedzi -------------------------------------------------


class CreditApplicationInput(BaseModel):
    """Dane wejściowe dla jednego klienta — odpowiadają polom formularza w app.py."""

    age: int = Field(ge=18, le=100, example=35)
    occupation: Literal[tuple(OCCUPATIONS)] = Field(example="Engineer")  # type: ignore[valid-type]
    annual_income: float = Field(ge=0, example=50000.0)
    monthly_salary: float = Field(ge=0, example=4000.0)
    monthly_balance: float = Field(ge=0, example=300.0)
    amount_invested: float = Field(ge=0, example=100.0)
    num_bank_accounts: int = Field(ge=0, le=20, example=4)
    num_credit_card: int = Field(ge=0, le=20, example=4)
    num_of_loan: int = Field(ge=0, le=20, example=2)
    interest_rate: float = Field(ge=0, le=50, example=12.0)
    outstanding_debt: float = Field(ge=0, example=1200.0)
    credit_utilization: float = Field(ge=0, le=100, example=32.0)
    total_emi: float = Field(ge=0, example=100.0)
    credit_mix: Literal["Bad", "Standard", "Good"] = Field(example="Standard")
    credit_history_age_months: int = Field(ge=0, example=120)
    delay_from_due_date: int = Field(ge=0, example=10)
    num_delayed_payment: int = Field(ge=0, example=5)
    changed_credit_limit: float = Field(example=8.0)
    num_credit_inquiries: int = Field(ge=0, example=4)
    payment_min: Literal["No", "Yes"] = Field(example="No")
    loan_types: list[Literal[tuple(LOAN_TYPES)]] = Field(default_factory=lambda: ["Personal Loan"])  # type: ignore[valid-type]
    payment_behaviour: Literal[tuple(PAYMENT_BEHAVIOURS)] = Field(  # type: ignore[valid-type]
        example="Low_spent_Small_value_payments"
    )


class PredictionResponse(BaseModel):
    request_id: str
    prediction_index: int
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float] | None = None
    verification_triggered: bool = False


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


class PredictionStats(BaseModel):
    total_predictions: int
    verify_every: int
    predictions_left_to_next_verification: int
    last_verification: dict[str, Any] | None = None


# Funkcja pomocnicza: jedna predykcja + logowanie ----------------------------


def _predict_one(payload: CreditApplicationInput, source: str) -> PredictionResponse:
    model = _get_model()
    inputs = payload.model_dump()

    features_df = inference.build_feature_row(inputs)
    preds, proba = inference.predict(model, features_df)

    predicted_class = int(preds[0])
    probabilities = None
    if proba is not None:
        classes = getattr(model, "classes_", [0, 1, 2])
        probabilities = {
            TARGET_LABELS.get(int(c), str(c)): float(proba[0][idx])
            for idx, c in enumerate(classes)
        }

    request_id = str(uuid.uuid4())
    record = prediction_logger.log_prediction(
        features=features_df.iloc[0].to_dict(),
        predicted_class=predicted_class,
        probabilities=probabilities,
        source=source,
        model_version=_app_state.get("model_version"),
        request_id=request_id,
    )

    return PredictionResponse(
        request_id=request_id,
        prediction_index=record["prediction_index"],
        predicted_class=predicted_class,
        predicted_label=TARGET_LABELS.get(predicted_class, str(predicted_class)),
        probabilities=probabilities,
        verification_triggered="verification" in record,
    )


# Endpointy --------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    model_ok = _app_state.get("model") is not None
    return {
        "status": "ok" if model_ok else "degraded",
        "model_loaded": model_ok,
        "model_version": _app_state.get("model_version"),
        "detail": None if model_ok else _app_state.get("model_error"),
    }


@app.get("/model/metrics")
def model_metrics() -> dict[str, Any]:
    if not inference.METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Plik metrics.json nie istnieje.")
    with open(inference.METRICS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@app.post("/predict", response_model=PredictionResponse)
def predict_single(payload: CreditApplicationInput) -> PredictionResponse:
    return _predict_one(payload, source="api")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payloads: list[CreditApplicationInput]) -> BatchPredictionResponse:
    if not payloads:
        raise HTTPException(status_code=400, detail="Lista wejściowa jest pusta.")
    results = [_predict_one(item, source="api_batch") for item in payloads]
    return BatchPredictionResponse(results=results)


@app.get("/predictions/stats", response_model=PredictionStats)
def predictions_stats() -> PredictionStats:
    return PredictionStats(**prediction_logger.stats())
