"""
eda.py
──────
Run this SECOND after generate_data.py.
Reads:   data/loan_applications.csv
         data/transactions.csv
Saves:   data/cleaned_loan_data.csv
         plots/loan_status_dist.png
         plots/correlation_map.png
         plots/cibil_distribution.png
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('plots', exist_ok=True)

print("=" * 55)
print("   🔍 LOAN GUARD AI — EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# ── 1. LOAD ────────────────────────────────────────────────────
apps = pd.read_csv('data/loan_applications.csv')
txns = pd.read_csv('data/transactions.csv')
print(f"\n✅ Applications : {apps.shape[0]:,} rows")
print(f"✅ Transactions : {txns.shape[0]:,} rows")

# ── 2. TRANSACTION FEATURE ENGINEERING ────────────────────────
txn_summary = txns.groupby('customer_id').agg(
    avg_spend        = ('transaction_amount', 'mean'),
    total_spend      = ('transaction_amount', 'sum'),
    txn_count        = ('transaction_amount', 'count'),
    past_fraud_count = ('fraud_flag', 'sum')
).reset_index()

# ── 3. MERGE ───────────────────────────────────────────────────
df = pd.merge(apps, txn_summary, on='customer_id', how='left')
# Fill numeric columns with 0, string columns with 'Unknown'
for col in df.columns:
    if df[col].dtype == 'object' or str(df[col].dtype) == 'string':
        df[col] = df[col].fillna('Unknown')
    else:
        df[col] = df[col].fillna(0)
print(f"✅ Merged shape : {df.shape}")

# ── 4. MISSING VALUES ──────────────────────────────────────────
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "None ✅")

# ── 5. FRAUD FLAG ──────────────────────────────────────────────
df['high_risk_flag'] = (df['past_fraud_count'] > 0).astype(int)
print(f"\n⚠️  High-risk applications : {df['high_risk_flag'].sum():,}")
print(f"✅ Low-risk  applications  : {(df['high_risk_flag'] == 0).sum():,}")

# ── 6. PLOT A — Loan Status Distribution ──────────────────────
plt.figure(figsize=(9, 5))
ax = sns.countplot(x='loan_status', data=df, palette='viridis',
                   order=df['loan_status'].value_counts().index)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)
plt.title('Loan Status Distribution', fontsize=14, fontweight='bold')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('plots/loan_status_dist.png', dpi=150)
plt.close()
print("\n✅ Saved → plots/loan_status_dist.png")

# ── 7. PLOT B — Correlation Heatmap ───────────────────────────
plt.figure(figsize=(14, 9))
num_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.4)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/correlation_map.png', dpi=150)
plt.close()
print("✅ Saved → plots/correlation_map.png")

# ── 8. PLOT C — CIBIL Score Distribution ──────────────────────
plt.figure(figsize=(10, 5))
for status in df['loan_status'].unique():
    df[df['loan_status'] == status]['cibil_score'].plot.kde(label=status)
plt.title('CIBIL Score Distribution by Loan Status', fontsize=13, fontweight='bold')
plt.xlabel('CIBIL Score')
plt.legend()
plt.tight_layout()
plt.savefig('plots/cibil_distribution.png', dpi=150)
plt.close()
print("✅ Saved → plots/cibil_distribution.png")

# ── 9. SAVE CLEANED DATA ──────────────────────────────────────
df.to_csv('data/cleaned_loan_data.csv', index=False)
print("✅ Saved → data/cleaned_loan_data.csv")
print("\n🎉 EDA complete! Now run: python train_model.py\n")
