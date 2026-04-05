"""
generate_data.py
─────────────────
Run this FIRST to create the two CSV files needed for the project.
Output:
  data/loan_applications.csv
  data/transactions.csv
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 10000  # number of records

os.makedirs('data', exist_ok=True)

# ── LOAN APPLICATIONS ──────────────────────────────────────────
customer_ids = [f'CUST{str(i).zfill(5)}' for i in range(1, N + 1)]

loan_types    = ['Home Loan', 'Car Loan', 'Personal Loan', 'Education Loan', 'Business Loan']
purposes      = ['Home Purchase', 'Debt Consolidation', 'Education', 'Medical', 'Travel', 'Business', 'Other']
emp_statuses  = ['Salaried', 'Self-Employed', 'Unemployed', 'Student', 'Retired']
prop_statuses = ['Owned', 'Rented', 'Mortgaged']
genders       = ['Male', 'Female', 'Other']

cibil_scores  = np.clip(np.random.normal(680, 80, N).astype(int), 300, 900)
incomes       = np.clip(np.random.normal(60000, 25000, N).astype(int), 10000, 300000)
loan_amounts  = np.clip(np.random.normal(300000, 150000, N).astype(int), 50000, 2000000)
tenures       = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 360], N)
existing_emis = np.clip(np.random.normal(5000, 4000, N).astype(int), 0, 50000)
ages          = np.clip(np.random.normal(38, 10, N).astype(int), 21, 65)
dependents    = np.random.randint(0, 5, N)
interest_rate = np.round(np.random.uniform(7.5, 14.5, N), 2)

dti = np.round((existing_emis + incomes * 0.3) / np.maximum(incomes, 1), 4)

# Derive loan_status from rules
def assign_status(cibil, income, loan_amt, emi, fraud_hist):
    if fraud_hist:
        return np.random.choice(
            ['Fraudulent - Detected', 'Fraudulent - Undetected'], p=[0.6, 0.4]
        )
    score = 0
    if cibil >= 750: score += 3
    elif cibil >= 650: score += 1
    else: score -= 2
    if income >= 50000: score += 2
    elif income >= 25000: score += 1
    else: score -= 1
    if loan_amt / max(income, 1) < 5: score += 1
    else: score -= 1
    if emi / max(income, 1) < 0.3: score += 1
    else: score -= 1
    return 'Approved' if score >= 3 else 'Declined'

fraud_hist_flags = np.random.choice([True, False], N, p=[0.08, 0.92])
loan_statuses = [
    assign_status(cibil_scores[i], incomes[i], loan_amounts[i],
                  existing_emis[i], fraud_hist_flags[i])
    for i in range(N)
]

apps_df = pd.DataFrame({
    'customer_id':               customer_ids,
    'application_id':            [f'APP{str(i).zfill(6)}' for i in range(1, N + 1)],
    'application_date':          pd.date_range('2022-01-01', periods=N, freq='1h').strftime('%Y-%m-%d'),
    'loan_type':                 np.random.choice(loan_types, N),
    'loan_amount_requested':     loan_amounts,
    'loan_tenure_months':        tenures,
    'interest_rate_offered':     interest_rate,
    'purpose_of_loan':           np.random.choice(purposes, N),
    'employment_status':         np.random.choice(emp_statuses, N),
    'monthly_income':            incomes,
    'cibil_score':               cibil_scores,
    'existing_emis_monthly':     existing_emis,
    'debt_to_income_ratio':      dti,
    'property_ownership_status': np.random.choice(prop_statuses, N),
    'applicant_age':             ages,
    'gender':                    np.random.choice(genders, N),
    'number_of_dependents':      dependents,
    'residential_address':       [f'City_{np.random.randint(1,50)}' for _ in range(N)],
    'loan_status':               loan_statuses,
    'fraud_type':                [
        np.random.choice(['Identity Theft', 'Income Falsification', 'None'])
        if 'Fraud' in s else 'None' for s in loan_statuses
    ],
})

apps_df.to_csv('data/loan_applications.csv', index=False)
print(f"✅ loan_applications.csv saved → {len(apps_df)} rows")

# ── TRANSACTIONS ───────────────────────────────────────────────
txn_rows = []
for i, cid in enumerate(customer_ids):
    n_txns = np.random.randint(5, 30)
    is_fraud_cust = fraud_hist_flags[i]
    for _ in range(n_txns):
        txn_rows.append({
            'customer_id':        cid,
            'transaction_amount': round(abs(np.random.normal(3000, 2000)), 2),
            'fraud_flag':         int(is_fraud_cust and np.random.rand() < 0.3),
        })

txns_df = pd.DataFrame(txn_rows)
txns_df.to_csv('data/transactions.csv', index=False)
print(f"✅ transactions.csv saved  → {len(txns_df)} rows")
print("\n🎉 Done! Now run: python eda.py")
