"""
app.py
──────
Run this LAST after train_model.py.
Loads everything from the models/ folder.
"""

import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Guard AI",
    page_icon="🏦",
    layout="wide",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)
st.markdown("""
<style>
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>""", unsafe_allow_html=True)

# ── LOAD FROM models/ FOLDER ───────────────────────────────────
@st.cache_resource
def load_all():
    all_models    = joblib.load('models/all_models.pkl')
    label_enc     = joblib.load('models/label_encoder.pkl')
    encoders      = joblib.load('models/encoders.pkl')
    model_columns = joblib.load('models/model_columns.pkl')
    with open('models/model_results.json') as f:
        results = json.load(f)
    with open('models/best_model_name.txt') as f:
        best_name = f.read().strip()
    return all_models, label_enc, encoders, model_columns, results, best_name

all_models, label_enc, encoders, model_columns, model_results, best_model_name = load_all()

STATUS_CLASSES = list(label_enc.classes_)

# ── HELPER — BUILD INPUT ROW ───────────────────────────────────
def build_input(fields: dict) -> pd.DataFrame:
    row = fields.copy()
    for col, le in encoders.items():
        if col in row and col != 'loan_status':
            row[col] = le.transform([str(row[col])])[0]
    return pd.DataFrame([row])[model_columns]

# ── HEADER ─────────────────────────────────────────────────────
st.title("🏦 Loan Guard AI — Eligibility & Fraud Detection")
st.markdown("Multi-model risk assessment powered by **5 machine learning classifiers**.")
st.divider()

tab1, tab2 = st.tabs(["🔍 Risk Assessment", "📊 Model Comparison"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    model_names   = list(all_models.keys())
    selected_name = st.sidebar.selectbox(
        "Choose Model",
        model_names,
        index=model_names.index(best_model_name)
    )
    st.sidebar.success(f"🏆 Best model: **{best_model_name}**")
    m = model_results[selected_name]
    st.sidebar.markdown("**Selected Model Metrics**")
    st.sidebar.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
    st.sidebar.metric("F1-Score", f"{m['f1_score']*100:.1f}%")
    st.sidebar.metric("CV Score", f"{m['cv_score']*100:.1f}%")

    # Form
    with st.form("loan_form"):
        st.subheader("Applicant Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**👤 Personal**")
            age        = st.number_input("Age", 18, 100, 30)
            gender     = st.selectbox("Gender", encoders['gender'].classes_)
            income     = st.number_input("Monthly Income (₹)", value=50000, step=1000)
            cibil      = st.number_input("CIBIL Score", 300, 900, 750)
            dependents = st.number_input("Dependents", 0, 10, 0)

        with c2:
            st.markdown("**💳 Loan**")
            loan_type    = st.selectbox("Loan Type",     encoders['loan_type'].classes_)
            loan_purpose = st.selectbox("Purpose",       encoders['purpose_of_loan'].classes_)
            amount       = st.number_input("Amount (₹)", value=200000, step=10000)
            tenure       = st.slider("Tenure (Months)", 12, 360, 36)

        with c3:
            st.markdown("**🏠 Financial**")
            emp_status    = st.selectbox("Employment",        encoders['employment_status'].classes_)
            prop_status   = st.selectbox("Property Ownership",encoders['property_ownership_status'].classes_)
            existing_emi  = st.number_input("Existing EMIs (₹)", value=0, step=500)
            monthly_spend = st.number_input("Monthly Spend (₹)", value=15000, step=1000)

        st.divider()
        past_fraud = st.radio("Known Fraud History?", ["No", "Yes"], horizontal=True)
        submitted  = st.form_submit_button(f"🚀 Assess with {selected_name}", use_container_width=True)

    if submitted:
        dti = ((existing_emi + monthly_spend) / income) if income > 0 else 0

        fields = {
            'loan_type':                 loan_type,
            'loan_amount_requested':     amount,
            'loan_tenure_months':        tenure,
            'interest_rate_offered':     10.5,
            'purpose_of_loan':           loan_purpose,
            'employment_status':         emp_status,
            'monthly_income':            income,
            'cibil_score':               cibil,
            'existing_emis_monthly':     existing_emi,
            'debt_to_income_ratio':      dti,
            'property_ownership_status': prop_status,
            'applicant_age':             age,
            'gender':                    gender,
            'number_of_dependents':      dependents,
            'fraud_flag':                1 if past_fraud == "Yes" else 0,
            'avg_spend':                 monthly_spend,
            'total_spend':               monthly_spend * 12,
            'txn_count':                 50,
            'past_fraud_count':          1 if past_fraud == "Yes" else 0,
            'high_risk_flag':            1 if past_fraud == "Yes" else 0,
        }

        input_df     = build_input(fields)
        model        = all_models[selected_name]
        prediction   = model.predict(input_df)[0]
        probs        = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
        status_label = label_enc.inverse_transform([prediction])[0]

        st.subheader("📋 Results")
        r1, r2 = st.columns(2)

        with r1:
            if 'Approved' in status_label and past_fraud == "No":
                st.success(f"✅ **{status_label.upper()}**")
                st.info("🟢 Risk Level: LOW")
                st.balloons()
            elif 'Fraud' in status_label or past_fraud == "Yes":
                st.error(f"🚨 **{status_label.upper()}**")
                st.warning("⚠️ High-risk / Fraud activity detected.")
            else:
                st.error(f"❌ **{status_label.upper()}**")
                st.warning("⚠️ Profile does not meet criteria.")

        with r2:
            st.metric("Debt-to-Income Ratio", f"{dti:.2f}",
                      delta="High" if dti > 0.5 else "Normal",
                      delta_color="inverse" if dti > 0.5 else "normal")
            st.metric("CIBIL Score", cibil,
                      delta="Good" if cibil >= 700 else "Low",
                      delta_color="normal" if cibil >= 700 else "inverse")

        if probs is not None:
            st.markdown("**Prediction Confidence**")
            prob_df = pd.DataFrame({'Status': STATUS_CLASSES, 'Prob': probs})
            for _, row in prob_df.sort_values('Prob', ascending=False).iterrows():
                st.progress(float(row['Prob']), text=f"{row['Status']}: {row['Prob']*100:.1f}%")

        st.caption(f"Model: **{selected_name}** | Accuracy: {model_results[selected_name]['accuracy']*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Model Performance Comparison")

    df_res = pd.DataFrame(model_results).T.reset_index()
    df_res.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score']
    df_res = df_res.sort_values('CV Score', ascending=False).reset_index(drop=True)

    def highlight(row):
        return ['background-color:#d4edda;font-weight:bold'
                if row['Model'] == best_model_name else '' for _ in row]

    st.dataframe(
        df_res.style.apply(highlight, axis=1).format({
            c: '{:.2%}' for c in ['Accuracy','Precision','Recall','F1-Score','CV Score']
        }),
        use_container_width=True, hide_index=True
    )
    st.caption(f"🏆 Best model (green): **{best_model_name}**")
    st.divider()

    # Bar chart
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.patch.set_facecolor('#0e1117')
    names   = df_res['Model'].tolist()
    colors  = ['#4CAF50' if n == best_model_name else '#2196F3' for n in names]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score']

    for ax, metric in zip(axes, metrics):
        vals = df_res[metric].values
        bars = ax.barh(names, vals, color=colors, edgecolor='white', linewidth=0.4)
        ax.set_xlim(0, 1.1)
        ax.set_title(metric, color='white', fontsize=10, fontweight='bold')
        ax.tick_params(colors='white', labelsize=7)
        ax.set_facecolor('#1a1a2e')
        for spine in ax.spines.values(): spine.set_visible(False)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{v:.0%}', va='center', color='white', fontsize=8)

    fig.legend(
        handles=[mpatches.Patch(color='#4CAF50', label=f'Best: {best_model_name}'),
                 mpatches.Patch(color='#2196F3', label='Other models')],
        loc='lower center', ncol=2, facecolor='#0e1117', labelcolor='white', fontsize=9
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    st.pyplot(fig)

    st.divider()
    st.subheader("🧠 Model Descriptions")
    info = {
        "Random Forest":       "Ensemble of 100 decision trees. Robust, handles mixed data well.",
        "Naive Bayes":         "Probabilistic model based on Bayes theorem. Very fast.",
        "Logistic Regression": "Linear classifier. Simple and highly interpretable.",
        "Decision Tree":       "Rule-based tree. Easy to explain, can overfit.",
        "Gradient Boosting":   "Sequential trees correcting errors. Often top accuracy.",
    }
    cols = st.columns(5)
    for col, (name, desc) in zip(cols, info.items()):
        with col:
            acc = model_results[name]['accuracy']
            st.markdown(f"**{'🏆' if name == best_model_name else '🤖'} {name}**")
            st.caption(desc)
            st.progress(acc, text=f"{acc*100:.1f}%")
