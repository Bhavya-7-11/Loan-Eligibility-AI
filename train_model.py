"""
train_model.py
──────────────
Run this THIRD after eda.py.
Reads:  data/cleaned_loan_data.csv
Saves:  models/loan_model.pkl          ← best model
        models/all_models.pkl          ← all 5 trained models
        models/label_encoder.pkl       ← encoder for loan_status
        models/encoders.pkl            ← encoders for all categories
        models/model_columns.pkl       ← feature column order
        models/model_results.json      ← accuracy of all 5 models
        models/best_model_name.txt     ← name of best model
"""

import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

os.makedirs('models', exist_ok=True)

print("=" * 60)
print("   🏦 LOAN GUARD AI — MULTI-MODEL TRAINING PIPELINE")
print("=" * 60)

# ── 1. LOAD ────────────────────────────────────────────────────
df = pd.read_csv('data/cleaned_loan_data.csv')
print(f"\n✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. DROP UNNECESSARY COLUMNS ────────────────────────────────
drop_cols = ['application_id', 'customer_id', 'application_date',
             'residential_address', 'fraud_type']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ── 3. ENCODE CATEGORIES ───────────────────────────────────────
encoders = {}
for col in df.select_dtypes(include='object').columns:
    if col == 'loan_status':
        continue
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Encode target
label_enc = LabelEncoder()
df['loan_status'] = label_enc.fit_transform(df['loan_status'].astype(str))
encoders['loan_status'] = label_enc

print(f"✅ Target classes : {list(label_enc.classes_)}")

# ── 4. SPLIT ───────────────────────────────────────────────────
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train : {len(X_train):,}  |  Test : {len(X_test):,}")

# ── 5. DEFINE 5 MODELS ─────────────────────────────────────────
models = {
    "Random Forest":      RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "Naive Bayes":        GaussianNB(),
    "Logistic Regression":LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Decision Tree":      DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
    "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}

# ── 6. TRAIN & EVALUATE ────────────────────────────────────────
print("\n" + "=" * 60)
print("   📊 TRAINING RESULTS")
print("=" * 60)

results       = {}
trained_models = {}

for name, model in models.items():
    print(f"\n🔄  {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        "cv_score":  round(float(cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()), 4),
    }
    trained_models[name] = model
    r = results[name]
    print(f"   Accuracy={r['accuracy']:.4f}  Precision={r['precision']:.4f}  "
          f"Recall={r['recall']:.4f}  F1={r['f1_score']:.4f}  CV={r['cv_score']:.4f}")

# ── 7. BEST MODEL ──────────────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]['cv_score'])
best_model = trained_models[best_name]
print(f"\n🏆  Best Model : {best_name}  (CV={results[best_name]['cv_score']:.4f})")

y_pred_best = best_model.predict(X_test)
print(f"\n{classification_report(y_test, y_pred_best, target_names=label_enc.classes_)}")

# ── 8. SAVE EVERYTHING TO models/ ──────────────────────────────
joblib.dump(best_model,          'models/loan_model.pkl')
joblib.dump(trained_models,      'models/all_models.pkl')
joblib.dump(label_enc,           'models/label_encoder.pkl')
joblib.dump(encoders,            'models/encoders.pkl')
joblib.dump(X.columns.tolist(),  'models/model_columns.pkl')

with open('models/model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
with open('models/best_model_name.txt', 'w') as f:
    f.write(best_name)

print("─" * 60)
print("✅  models/loan_model.pkl")
print("✅  models/all_models.pkl")
print("✅  models/label_encoder.pkl")
print("✅  models/encoders.pkl")
print("✅  models/model_columns.pkl")
print("✅  models/model_results.json")
print("✅  models/best_model_name.txt")
print("─" * 60)
print("\n🎉  Training complete! Now run: streamlit run app.py\n")
