from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ścieżki projektu
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "data" / "06_models" / "baseline_random_forest.pkl"
METRICS_PATH = PROJECT_ROOT / "data" / "08_reporting" / "metrics.json"


sys.path.insert(0, str(PROJECT_ROOT / "src"))


# Definicje cech
TARGET_LABELS = {0: "Poor", 1: "Standard", 2: "Good"}
TARGET_PL = {0: "Niska (Poor)", 1: "Średnia (Standard)", 2: "Dobra (Good)"}
DEFAULT_COLOR = {0: "#35A4E5", 1: "#8378FF"}
TARGET_COLOR = {0: "#e53935", 1: "#fb8c00", 2: "#43a047"}

LOAN_TYPES = [
    "Auto Loan",
    "Credit-Builder Loan",
    "Debt Consolidation Loan",
    "Home Equity Loan",
    "Mortgage Loan",
    "Not Specified",
    "Payday Loan",
    "Personal Loan",
    "Student Loan",
]

PAYMENT_BEHAVIOURS = [
    "High_spent_Large_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments",
    "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Small_value_payments",
]

OCCUPATIONS = [
    "Architect",
    "Developer",
    "Doctor",
    "Engineer",
    "Entrepreneur",
    "Journalist",
    "Lawyer",
    "Manager",
    "Mechanic",
    "Media_Manager",
    "Musician",
    "Scientist",
    "Teacher",
    "Writer",
]

CREDIT_MIX_MAP = {"Bad": 0, "Standard": 1, "Good": 2}
PAYMENT_MIN_MAP = {"No": 0, "Yes": 1}

# Kolejność cech wymagana przez model — identyczna jak przy treningu.
MODEL_FEATURES = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    *[f"LoanType_{lt}" for lt in LOAN_TYPES],
    *[f"PayBeh_{pb}" for pb in PAYMENT_BEHAVIOURS],
    *[f"Occupation_{oc}" for oc in OCCUPATIONS],
]


# Ładowanie modelu i metryk
def _is_lfs_pointer(path: Path) -> bool:
    """Wykrywa, czy plik to wskaźnik Git LFS, a nie faktyczny model."""
    try:
        if path.stat().st_size > 5000:  # ograniczenie pkl (setki MB)
            return False
        with open(path, "rb") as handle:
            head = handle.read(200)
        return b"git-lfs" in head
    except OSError:
        return False


@st.cache_resource(show_spinner="Wczytywanie modelu...")
def load_model(path_str: str):
    """Wczytuje wytrenowany model z pliku pickle (cache na czas sesji)."""
    with open(path_str, "rb") as handle:
        return pickle.load(handle)


@st.cache_data(show_spinner=False)
def load_metrics(path_str: str) -> dict:
    try:
        with open(path_str, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


# Budowa wektora cech z formularza
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
    if inputs["occupation"] in OCCUPATIONS:
        row[f"Occupation_{inputs['occupation']}"] = 1

    return pd.DataFrame([row])[MODEL_FEATURES]


# Predykcja
def predict(model, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    preds = model.predict(features)
    proba = model.predict_proba(features) if hasattr(model, "predict_proba") else None
    return preds, proba


# UI
st.set_page_config(
    page_title="Predykcja oceny kredytowej",
    page_icon="💳",
    layout="wide",
)

colormain = DEFAULT_COLOR.get(1)

st.markdown(
    """
    <style>
    /* ---- Margines od boków okna + ograniczona szerokość treści ---- */

    .block-container {
        max-width: 1200px;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 2.5rem;
        margin: 0 auto;
    }

    /* Mniejsze, bardziej zwarte pola wejściowe */
    div[data-testid="stNumberInput"],
    div[data-testid="stSelectbox"],
    div[data-testid="stMultiSelect"],
    div[data-testid="stTextInput"] {
        max-width: 800px;
    }
 

    /* ------------ Przyciski ------------ */
    /* ---- Przycisk główny (type="primary") ---- */

    div[data-testid="stButton"] > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #8378FF22;
        color: #6B5CFF;
        border: 1px solid #8378FF;
        border-radius: 8px;
        font-weight: 600;
        transition: background-color 0.15s ease, transform 0.05s ease,
            box-shadow 0.15s ease;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #8378FF44;
        border-color: #8378FF;
        color: #FFFFFF;
    }

    div[data-testid="stButton"] > button[kind="primary"]:active,
    button[data-testid="stBaseButton-primary"]:active {
        transform: translateY(1px);
    }
 

    /* ---- Przyciski drugorzędne i przycisk pobierania ---- */

    div[data-testid="stButton"] > button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"],
    div[data-testid="stDownloadButton"] > button,
    button[data-testid="stBaseButton-secondaryFormSubmit"] {
        background-color: #ffffff;
        color: #4f46e5;
        border: 1px solid #4f46e5;
        border-radius: 8px;
        font-weight: 600;
        transition: background-color 0.15s ease, color 0.15s ease;
    }

    div[data-testid="stButton"] > button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
        background-color: #eef2ff;
        color: #8378FF;
        border-color: #8378FF;
    }


    /* ---- Uploader plików na ciemno (przycisk "Upload" + dropzone) ---- */
    
    section[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #1c2230 !important;
        border: 1px dashed #2c3444 !important;
    }

    div[data-testid="stFileUploader"] button,
    section[data-testid="stFileUploaderDropzone"] button {
        background-color: #8378FF22 !important;
        color: #6B5CFF !important;
        border: 1px solid #8378FF !important;
    }

    div[data-testid="stFileUploader"] button:hover,
    section[data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #8378FF44 !important;
        color: #ffffff !important;
        border-color: #8378FF !important;
    }

    div[data-testid="stFileUploaderDropzoneInstructions"],
    div[data-testid="stFileUploaderDropzoneInstructions"] span,
    div[data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #b8bdc9 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.title("💳 Predykcja oceny kredytowej")
st.caption(
    "Model Random Forest (projekt ASI / Kedro) klasyfikujący ocenę kredytową "
    "klienta jako **Poor**, **Standard** lub **Good**."
)

# Sprawdzenie dostępności modelu #
if not MODEL_PATH.exists():
    st.error(f"Nie znaleziono pliku modelu: `{MODEL_PATH}`")
    st.stop()

if _is_lfs_pointer(MODEL_PATH):
    st.warning(
        "⚠️ Plik modelu to tylko **wskaźnik Git LFS**, a nie faktyczny model.\n\n"
        "Pobierz prawdziwy model poleceniem:\n\n"
        "```bash\ngit lfs pull\n```\n\n"
        "Alternatywnie uruchom pipeline treningowy Kedro: `kedro run --pipeline=modeling`."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:  # noqa: BLE001
    st.error(f"Nie udało się wczytać modelu: {exc}")
    st.stop()


def render_result(pred: int, proba: np.ndarray | None) -> None:
    """Wyświetla wynik predykcji wraz z rozkładem prawdopodobieństw."""
    color = TARGET_COLOR.get(pred, "#607d8b")
    label = TARGET_PL.get(pred, str(pred))
    st.markdown(
        f"""
        <div style="padding:1.2rem;border-radius:12px;background:{color}22;
                    border:2px solid {color};text-align:center;">
            <div style="font-size:0.9rem;color:#555;">Przewidywana ocena kredytowa</div>
            <div style="font-size:2rem;font-weight:700;color:{color};">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if proba is not None:
        classes = getattr(model, "classes_", [0, 1, 2])
        proba_df = pd.DataFrame(
            {
                "Klasa": [TARGET_LABELS.get(int(c), str(c)) for c in classes],
                "Prawdopodobieństwo": proba[0],
            }
        ).set_index("Klasa")
        st.bar_chart(proba_df, height=220)


tab_single, tab_batch = st.tabs(["Pojedyncza predykcja", "Predykcja wsadowa (CSV)"])

# TAB — pojedyncza predykcja
with tab_single:
    st.subheader("Wprowadź dane klienta")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Dane podstawowe**")
        age = st.number_input("Wiek", min_value=18, max_value=100, value=35)
        occupation = st.selectbox("Zawód", OCCUPATIONS, index=3)
        annual_income = st.number_input(
            "Roczny dochód", min_value=0.0, value=50000.0, step=1000.0
        )
        monthly_salary = st.number_input(
            "Miesięczne wynagrodzenie netto", min_value=0.0, value=4000.0, step=100.0
        )
        monthly_balance = st.number_input(
            "Miesięczne saldo", min_value=0.0, value=300.0, step=50.0
        )
        amount_invested = st.number_input(
            "Miesięczna kwota inwestycji", min_value=0.0, value=100.0, step=10.0
        )

    with col2:
        st.markdown("**Konta i kredyty**")
        num_bank_accounts = st.number_input(
            "Liczba kont bankowych", min_value=0, max_value=20, value=4
        )
        num_credit_card = st.number_input(
            "Liczba kart kredytowych", min_value=0, max_value=20, value=4
        )
        num_of_loan = st.number_input(
            "Liczba kredytów", min_value=0, max_value=20, value=2
        )
        interest_rate = st.number_input(
            "Oprocentowanie (%)", min_value=0.0, max_value=50.0, value=12.0
        )
        outstanding_debt = st.number_input(
            "Zadłużenie pozostałe", min_value=0.0, value=1200.0, step=100.0
        )
        credit_utilization = st.number_input(
            "Wskaźnik wykorzystania kredytu (%)",
            min_value=0.0,
            max_value=100.0,
            value=32.0,
        )
        total_emi = st.number_input(
            "Łączna miesięczna rata (EMI)", min_value=0.0, value=100.0, step=10.0
        )

    with col3:
        st.markdown("**Historia płatności**")
        credit_mix = st.selectbox(
            "Jakość portfela kredytowego (Credit Mix)",
            list(CREDIT_MIX_MAP.keys()),
            index=1,
        )
        credit_history_years = st.number_input(
            "Historia kredytowa — lata", min_value=0, max_value=60, value=10
        )
        credit_history_extra_months = st.number_input(
            "Historia kredytowa — dodatkowe miesiące",
            min_value=0,
            max_value=11,
            value=0,
        )
        delay_from_due_date = st.number_input(
            "Średnie opóźnienie od terminu (dni)", min_value=0, value=10
        )
        num_delayed_payment = st.number_input(
            "Liczba opóźnionych płatności", min_value=0, value=5
        )
        changed_credit_limit = st.number_input(
            "Zmiana limitu kredytowego", value=8.0, step=1.0
        )
        num_credit_inquiries = st.number_input(
            "Liczba zapytań kredytowych", min_value=0, value=4
        )
        payment_min = st.selectbox(
            "Spłata minimalnej kwoty", list(PAYMENT_MIN_MAP.keys()), index=0
        )

    st.markdown("**Profil kredytowy i płatniczy**")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        loan_types = st.multiselect(
            "Typy posiadanych kredytów",
            LOAN_TYPES,
            default=["Personal Loan"],
        )
    with pcol2:
        payment_behaviour = st.selectbox(
            "Zachowanie płatnicze", PAYMENT_BEHAVIOURS, index=1
        )

    if st.button("Przewidź", type="primary", use_container_width=True):
        inputs = {
            "age": age,
            "occupation": occupation,
            "annual_income": annual_income,
            "monthly_salary": monthly_salary,
            "monthly_balance": monthly_balance,
            "amount_invested": amount_invested,
            "num_bank_accounts": num_bank_accounts,
            "num_credit_card": num_credit_card,
            "num_of_loan": num_of_loan,
            "interest_rate": interest_rate,
            "outstanding_debt": outstanding_debt,
            "credit_utilization": credit_utilization,
            "total_emi": total_emi,
            "credit_mix": credit_mix,
            "credit_history_age_months": credit_history_years * 12 + credit_history_extra_months,
            "delay_from_due_date": delay_from_due_date,
            "num_delayed_payment": num_delayed_payment,
            "changed_credit_limit": changed_credit_limit,
            "num_credit_inquiries": num_credit_inquiries,
            "payment_min": payment_min,
            "loan_types": loan_types if loan_types else ["Not Specified"],
            "payment_behaviour": payment_behaviour,
        }
        features = build_feature_row(inputs)
        preds, proba = predict(model, features)
        st.divider()
        render_result(int(preds[0]), proba)

        with st.expander("Pokaż wektor cech przekazany do modelu"):
            st.dataframe(features.T.rename(columns={0: "wartość"}))
    else:
        color = DEFAULT_COLOR.get(0)
        st.divider()
        st.markdown(
            f"""
            <div style="padding:1.2rem;border-radius:12px;background: {color}22;
                        border:2px solid {color};text-align:center;">
                <div style="font-size:0.9rem;color:#555;">Tutaj pojawi się ocena kredytowa</div>
                <div style="font-size:1.5rem;font-weight:700;color: {color};">Wprowadź dane i kliknij przycisk "przewidź"</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# TAB — predykcja wsadowa
with tab_batch:
    st.subheader("Predykcja dla pliku CSV")
    st.markdown(
        "Wgraj plik CSV. Obsługiwane są dwa formaty:\n"
        "- **dane surowe** w formacie `credit_score.csv` (zostaną przetworzone "
        "tym samym pipeline'em co trening),\n"
        "- **dane gotowe** zawierające wszystkie kolumny modelu (`MODEL_FEATURES`)."
    )

    uploaded = st.file_uploader("Plik CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Nie udało się wczytać pliku CSV: {exc}")
            st.stop()

        st.write(f"Wczytano **{len(raw_df)}** wierszy, {raw_df.shape[1]} kolumn.")
        st.dataframe(raw_df.head(), use_container_width=True)

        has_all_features = all(col in raw_df.columns for col in MODEL_FEATURES)

        model_input = None
        if has_all_features:
            st.info("Wykryto gotowe kolumny modelu — pomijam przetwarzanie.")
            model_input = raw_df[MODEL_FEATURES].copy()
        else:
            st.info("Dane surowe — uruchamiam pipeline przetwarzania (Kedro).")
            try:
                from credit_scoring.pipelines.data_processing.nodes import (
                    clean_credit_data,
                    prepare_model_input,
                )

                cleaned = clean_credit_data(raw_df)
                prepared = prepare_model_input(cleaned)
                model_input = prepared.drop(columns=["Credit_Score"], errors="ignore")
                model_input = model_input[MODEL_FEATURES]
            except Exception as exc:  # noqa: BLE001
                st.error(
                    "Nie udało się przetworzyć danych surowych. "
                    f"Sprawdź, czy CSV ma format `credit_score.csv`.\n\nSzczegóły: {exc}"
                )
                st.stop()

        if st.button("Przewiduj dla całego pliku", type="primary"):
            preds, proba = predict(model, model_input)
            result = raw_df.copy()
            result["Predykcja"] = [TARGET_LABELS.get(int(p), str(p)) for p in preds]
            if proba is not None:
                classes = getattr(model, "classes_", [0, 1, 2])
                for idx, cls in enumerate(classes):
                    result[f"P_{TARGET_LABELS.get(int(cls), cls)}"] = proba[:, idx].round(4)

            st.success("Gotowe!")
            st.dataframe(result.head(50), use_container_width=True)

            counts = pd.Series(
                [TARGET_LABELS.get(int(p), str(p)) for p in preds]
            ).value_counts()
            st.bar_chart(counts)

            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Pobierz wyniki (CSV)",
                data=csv_bytes,
                file_name="predykcje_credit_score.csv",
                mime="text/csv",
            )