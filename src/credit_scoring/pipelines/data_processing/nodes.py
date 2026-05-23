import re

import numpy as np
import pandas as pd


TARGET_MAP = {"Poor": 0, "Standard": 1, "Good": 2}

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
    "LoanType_Auto Loan",
    "LoanType_Credit-Builder Loan",
    "LoanType_Debt Consolidation Loan",
    "LoanType_Home Equity Loan",
    "LoanType_Mortgage Loan",
    "LoanType_Not Specified",
    "LoanType_Payday Loan",
    "LoanType_Personal Loan",
    "LoanType_Student Loan",
    "PayBeh_High_spent_Large_value_payments",
    "PayBeh_High_spent_Medium_value_payments",
    "PayBeh_High_spent_Small_value_payments",
    "PayBeh_Low_spent_Large_value_payments",
    "PayBeh_Low_spent_Medium_value_payments",
    "PayBeh_Low_spent_Small_value_payments",
    "Occupation_Architect",
    "Occupation_Developer",
    "Occupation_Doctor",
    "Occupation_Engineer",
    "Occupation_Entrepreneur",
    "Occupation_Journalist",
    "Occupation_Lawyer",
    "Occupation_Manager",
    "Occupation_Mechanic",
    "Occupation_Media_Manager",
    "Occupation_Musician",
    "Occupation_Scientist",
    "Occupation_Teacher",
    "Occupation_Writer",
]


def _clean_numeric(series: pd.Series, allow_negative: bool = False) -> pd.Series:
    pattern = r"[^0-9.\-]" if allow_negative else r"[^0-9.]"
    cleaned = series.astype(str).str.replace(pattern, "", regex=True)
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def _fill_numeric_by_customer(data: pd.DataFrame, column: str) -> pd.Series:
    if "Customer_ID" in data.columns:
        filled = data.groupby("Customer_ID")[column].transform(lambda x: x.ffill().bfill())
    else:
        filled = data[column]

    median = filled.median()
    return filled.fillna(median)


def _fill_categorical_by_customer(
    data: pd.DataFrame, column: str, default: str
) -> pd.Series:
    if "Customer_ID" in data.columns:
        filled = data.groupby("Customer_ID")[column].transform(lambda x: x.ffill().bfill())
    else:
        filled = data[column]

    mode = filled.mode(dropna=True)
    fallback = mode.iloc[0] if not mode.empty else default
    return filled.fillna(fallback)


def _parse_credit_history_age(value: object) -> float:
    if pd.isna(value) or str(value).upper() == "NA":
        return np.nan

    match = re.search(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", str(value))
    if match is None:
        return np.nan

    years, months = match.groups()
    return int(years) * 12 + int(months)


def _add_loan_type_features(data: pd.DataFrame) -> pd.DataFrame:
    loan_values = data["Type_of_Loan"].fillna("Not Specified").astype(str)

    for loan_type in LOAN_TYPES:
        data[f"LoanType_{loan_type}"] = loan_values.str.contains(
            re.escape(loan_type), case=False, na=False
        ).astype(int)

    return data


def _add_payment_behaviour_features(data: pd.DataFrame) -> pd.DataFrame:
    data["Payment_Behaviour"] = data["Payment_Behaviour"].replace("!@9#%8", np.nan)
    data["Payment_Behaviour"] = _fill_categorical_by_customer(
        data, "Payment_Behaviour", PAYMENT_BEHAVIOURS[0]
    )

    for behaviour in PAYMENT_BEHAVIOURS:
        data[f"PayBeh_{behaviour}"] = (
            data["Payment_Behaviour"] == behaviour
        ).astype(int)

    return data


def clean_credit_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()

    cleaned = cleaned.drop(columns=["ID", "Name", "SSN"], errors="ignore")

    cleaned["Monthly_Inhand_Salary"] = _clean_numeric(
        cleaned["Monthly_Inhand_Salary"]
    )
    cleaned["Monthly_Inhand_Salary"] = _fill_numeric_by_customer(
        cleaned, "Monthly_Inhand_Salary"
    )

    cleaned["Age"] = _clean_numeric(cleaned["Age"], allow_negative=True)
    cleaned.loc[(cleaned["Age"] < 0) | (cleaned["Age"] > 100), "Age"] = np.nan
    cleaned["Age"] = _fill_numeric_by_customer(cleaned, "Age")

    cleaned["Num_Bank_Accounts"] = _clean_numeric(
        cleaned["Num_Bank_Accounts"], allow_negative=True
    )
    cleaned.loc[cleaned["Num_Bank_Accounts"] < 0, "Num_Bank_Accounts"] = np.nan
    cleaned.loc[cleaned["Num_Bank_Accounts"] > 10, "Num_Bank_Accounts"] = np.nan
    cleaned["Num_Bank_Accounts"] = _fill_numeric_by_customer(
        cleaned, "Num_Bank_Accounts"
    )

    cleaned["Num_Credit_Card"] = _clean_numeric(cleaned["Num_Credit_Card"])
    cleaned.loc[cleaned["Num_Credit_Card"] > 13, "Num_Credit_Card"] = np.nan
    cleaned["Num_Credit_Card"] = _fill_numeric_by_customer(
        cleaned, "Num_Credit_Card"
    )

    cleaned["Interest_Rate"] = _clean_numeric(cleaned["Interest_Rate"])
    cleaned.loc[cleaned["Interest_Rate"] > 40, "Interest_Rate"] = np.nan
    cleaned["Interest_Rate"] = _fill_numeric_by_customer(cleaned, "Interest_Rate")

    cleaned["Num_of_Loan"] = _clean_numeric(cleaned["Num_of_Loan"], allow_negative=True)
    cleaned.loc[cleaned["Num_of_Loan"] < 0, "Num_of_Loan"] = np.nan
    cleaned.loc[cleaned["Num_of_Loan"] > 15, "Num_of_Loan"] = np.nan
    cleaned["Num_of_Loan"] = _fill_numeric_by_customer(cleaned, "Num_of_Loan")

    cleaned["Num_of_Delayed_Payment"] = _clean_numeric(
        cleaned["Num_of_Delayed_Payment"], allow_negative=True
    )
    cleaned.loc[
        cleaned["Num_of_Delayed_Payment"] > 100, "Num_of_Delayed_Payment"
    ] = np.nan
    cleaned["Num_of_Delayed_Payment"] = _fill_numeric_by_customer(
        cleaned, "Num_of_Delayed_Payment"
    )

    cleaned["Delay_from_due_date"] = _clean_numeric(
        cleaned["Delay_from_due_date"], allow_negative=True
    )
    cleaned.loc[cleaned["Delay_from_due_date"] < 0, "Delay_from_due_date"] = np.nan
    cleaned["Delay_from_due_date"] = _fill_numeric_by_customer(
        cleaned, "Delay_from_due_date"
    )

    cleaned["Annual_Income"] = _clean_numeric(cleaned["Annual_Income"])
    income_upper_bound = cleaned["Annual_Income"].quantile(0.99)
    cleaned.loc[cleaned["Annual_Income"] > income_upper_bound, "Annual_Income"] = np.nan
    cleaned["Annual_Income"] = _fill_numeric_by_customer(cleaned, "Annual_Income")

    cleaned["Occupation"] = cleaned["Occupation"].replace("_______", np.nan)
    cleaned["Occupation"] = _fill_categorical_by_customer(
        cleaned, "Occupation", "Accountant"
    )

    cleaned["Changed_Credit_Limit"] = _clean_numeric(
        cleaned["Changed_Credit_Limit"], allow_negative=True
    )
    cleaned["Changed_Credit_Limit"] = _fill_numeric_by_customer(
        cleaned, "Changed_Credit_Limit"
    )

    cleaned["Num_Credit_Inquiries"] = _clean_numeric(cleaned["Num_Credit_Inquiries"])
    cleaned.loc[
        cleaned["Num_Credit_Inquiries"] > 50, "Num_Credit_Inquiries"
    ] = np.nan
    cleaned["Num_Credit_Inquiries"] = _fill_numeric_by_customer(
        cleaned, "Num_Credit_Inquiries"
    )

    cleaned["Outstanding_Debt"] = _clean_numeric(cleaned["Outstanding_Debt"])
    cleaned["Outstanding_Debt"] = _fill_numeric_by_customer(
        cleaned, "Outstanding_Debt"
    )

    cleaned["Credit_Utilization_Ratio"] = _clean_numeric(
        cleaned["Credit_Utilization_Ratio"]
    )
    cleaned["Credit_Utilization_Ratio"] = _fill_numeric_by_customer(
        cleaned, "Credit_Utilization_Ratio"
    )

    cleaned["Credit_History_Age"] = cleaned["Credit_History_Age"].apply(
        _parse_credit_history_age
    )
    cleaned["Credit_History_Age"] = _fill_numeric_by_customer(
        cleaned, "Credit_History_Age"
    )

    cleaned["Credit_Mix"] = cleaned["Credit_Mix"].replace("_", np.nan)
    cleaned["Credit_Mix"] = _fill_categorical_by_customer(
        cleaned, "Credit_Mix", "Standard"
    )
    cleaned["Credit_Mix"] = cleaned["Credit_Mix"].map(
        {"Bad": 0, "Standard": 1, "Good": 2}
    )
    cleaned["Credit_Mix"] = cleaned["Credit_Mix"].fillna(
        cleaned["Credit_Mix"].median()
    )

    cleaned["Payment_of_Min_Amount"] = cleaned["Payment_of_Min_Amount"].replace(
        "NM", np.nan
    )
    cleaned["Payment_of_Min_Amount"] = _fill_categorical_by_customer(
        cleaned, "Payment_of_Min_Amount", "No"
    )
    cleaned["Payment_of_Min_Amount"] = cleaned["Payment_of_Min_Amount"].map(
        {"No": 0, "Yes": 1}
    )

    cleaned["Total_EMI_per_month"] = _clean_numeric(cleaned["Total_EMI_per_month"])
    emi_upper_bound = cleaned["Total_EMI_per_month"].quantile(0.99)
    cleaned.loc[
        cleaned["Total_EMI_per_month"] > emi_upper_bound, "Total_EMI_per_month"
    ] = emi_upper_bound
    cleaned["Total_EMI_per_month"] = _fill_numeric_by_customer(
        cleaned, "Total_EMI_per_month"
    )

    cleaned["Amount_invested_monthly"] = _clean_numeric(
        cleaned["Amount_invested_monthly"]
    )
    cleaned["Amount_invested_monthly"] = _fill_numeric_by_customer(
        cleaned, "Amount_invested_monthly"
    )

    cleaned["Monthly_Balance"] = _clean_numeric(
        cleaned["Monthly_Balance"], allow_negative=True
    )
    cleaned.loc[cleaned["Monthly_Balance"] < 0, "Monthly_Balance"] = np.nan
    cleaned["Monthly_Balance"] = _fill_numeric_by_customer(
        cleaned, "Monthly_Balance"
    )

    cleaned = _add_loan_type_features(cleaned)
    cleaned = _add_payment_behaviour_features(cleaned)

    cleaned["Credit_Score"] = cleaned["Credit_Score"].map(TARGET_MAP)

    return cleaned


def prepare_model_input(cleaned: pd.DataFrame) -> pd.DataFrame:
    model_input = cleaned.copy()

    for occupation in OCCUPATIONS:
        model_input[f"Occupation_{occupation}"] = (
            model_input["Occupation"] == occupation
        ).astype(int)

    for feature in MODEL_FEATURES:
        if feature not in model_input.columns:
            model_input[feature] = 0

    output_columns = MODEL_FEATURES + ["Credit_Score"]
    model_input = model_input[output_columns].copy()
    model_input = model_input.replace([np.inf, -np.inf], np.nan)

    for feature in MODEL_FEATURES:
        if model_input[feature].isna().any():
            model_input[feature] = model_input[feature].fillna(
                model_input[feature].median()
            )

    model_input = model_input.dropna(subset=["Credit_Score"])
    model_input["Credit_Score"] = model_input["Credit_Score"].astype(int)

    return model_input
