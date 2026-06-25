"""Wczytywanie modelu oraz budowa wektora cech / predykcja."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_scoring.serving.schema import (
    CREDIT_MIX_MAP,
    MODEL_FEATURES,
    PAYMENT_MIN_MAP,
)

# Korzeń projektu Kedro: src/credit_scoring/serving/inference.py -> 3 poziomy wyżej
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "data" / "06_models" / "baseline_random_forest.pkl"
METRICS_PATH = PROJECT_ROOT / "data" / "08_reporting" / "metrics.json"


class ModelNotAvailableError(RuntimeError):
    """Zgłaszany, gdy plik modelu nie istnieje lub jest jedynie wskaźnikiem Git LFS."""


def is_lfs_pointer(path: Path) -> bool:
    """Wykrywa, czy plik to wskaźnik Git LFS, a nie faktyczny model."""
    try:
        if path.stat().st_size > 5000:  # ograniczenie pkl (setki MB)
            return False
        with open(path, "rb") as handle:
            head = handle.read(200)
        return b"git-lfs" in head
    except OSError:
        return False


def load_model(path: Path | str = MODEL_PATH) -> Any:
    """Wczytuje wytrenowany model z pliku pickle.

    Nie używa cache'owania samodzielnie — w Streamlit owijamy tę funkcję
    w `@st.cache_resource`, a w FastAPI model wczytujemy raz przy starcie
    aplikacji (zob. `credit_scoring.serving.api`).
    """
    path = Path(path)
    if not path.exists():
        raise ModelNotAvailableError(f"Nie znaleziono pliku modelu: {path}")
    if is_lfs_pointer(path):
        raise ModelNotAvailableError(
            f"Plik modelu '{path}' to tylko wskaźnik Git LFS. "
            "Uruchom `git lfs pull` lub `kedro run --pipeline=modeling`."
        )
    with open(path, "rb") as handle:
        return pickle.load(handle)


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Tworzy pojedynczy wiersz cech w kolejności wymaganej przez model."""
    row = {feature: 0 for feature in MODEL_FEATURES}

    # Cechy numeryczne i porządkowe
    row["Age"] = inputs["age"]
    row["Annual_Income"] = inputs["annual_income"]
    row["Monthly_Inhand_Salary"] = inputs["monthly_salary"]
    row["Num_Bank_Accounts"] = inputs["num_bank_accounts"]
    row["Num_Credit_Card"] = inputs["num_credit_card"]
    row["Interest_Rate"] = inputs["interest_rate"]
    row["Num_of_Loan"] = inputs["num_of_loan"]
    row["Delay_from_due_date"] = inputs["delay_from_due_date"]
    row["Num_of_Delayed_Payment"] = inputs["num_delayed_payment"]
    row["Changed_Credit_Limit"] = inputs["changed_credit_limit"]
    row["Num_Credit_Inquiries"] = inputs["num_credit_inquiries"]
    row["Credit_Mix"] = CREDIT_MIX_MAP[inputs["credit_mix"]]
    row["Outstanding_Debt"] = inputs["outstanding_debt"]
    row["Credit_Utilization_Ratio"] = inputs["credit_utilization"]
    row["Credit_History_Age"] = inputs["credit_history_age_months"]
    row["Payment_of_Min_Amount"] = PAYMENT_MIN_MAP[inputs["payment_min"]]
    row["Total_EMI_per_month"] = inputs["total_emi"]
    row["Amount_invested_monthly"] = inputs["amount_invested"]
    row["Monthly_Balance"] = inputs["monthly_balance"]

    # Typy kredytów (multiselect -> one-hot)
    for loan_type in inputs["loan_types"]:
        row[f"LoanType_{loan_type}"] = 1

    # Zachowanie płatnicze (one-hot)
    row[f"PayBeh_{inputs['payment_behaviour']}"] = 1

    # Zawód (one-hot)
    occupation_keys = {
        key.split("Occupation_", 1)[1]
        for key in MODEL_FEATURES
        if key.startswith("Occupation_")
    }
    if inputs["occupation"] in occupation_keys:
        row[f"Occupation_{inputs['occupation']}"] = 1

    return pd.DataFrame([row])[MODEL_FEATURES]


def predict(model: Any, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Wykonuje predykcję klasy oraz (jeśli dostępne) prawdopodobieństw."""
    preds = model.predict(features)
    proba = model.predict_proba(features) if hasattr(model, "predict_proba") else None
    return preds, proba
