"""Definicje cech i słowniki kodowania używane przy predykcji."""
from __future__ import annotations

TARGET_LABELS: dict[int, str] = {0: "Poor", 1: "Standard", 2: "Good"}
TARGET_PL: dict[int, str] = {
    0: "Niska (Poor)",
    1: "Średnia (Standard)",
    2: "Dobra (Good)",
}

LOAN_TYPES: list[str] = [
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

PAYMENT_BEHAVIOURS: list[str] = [
    "High_spent_Large_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments",
    "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Small_value_payments",
]

OCCUPATIONS: list[str] = [
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

CREDIT_MIX_MAP: dict[str, int] = {"Bad": 0, "Standard": 1, "Good": 2}
PAYMENT_MIN_MAP: dict[str, int] = {"No": 0, "Yes": 1}

# Kolejność cech wymagana przez model — identyczna jak przy treningu.
MODEL_FEATURES: list[str] = [
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
