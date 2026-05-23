import pandas as pd


def clean_credit_data(data: pd.DataFrame) -> pd.DataFrame:

    cleaned = data.copy()
    cleaned = cleaned.drop(columns=["ID", "Name", "SSN"], errors="ignore")
    cleaned["Credit_Score"] = cleaned["Credit_Score"].map(
        {"Poor": 0, "Standard": 1, "Good": 2}
    )

    return cleaned


def prepare_model_input(cleaned: pd.DataFrame) -> pd.DataFrame:

    model_input = cleaned.copy()
    model_input = model_input.drop(columns=["Customer_ID", "Month"], errors="ignore")

    return model_input