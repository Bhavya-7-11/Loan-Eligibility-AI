# 🏦 Loan Guard AI




## ▶️ Run Order — Step by Step

Open your terminal, go into the folder, then run in this exact order:

```bash
# Go into your folder
cd path/to/loan_guard_ai

# Step 0 — Install libraries (only once, ever)
pip install -r requirements.txt

# Step 1 — Create the data
python generate_data.py

# Step 2 — Clean data + save plots
python eda.py

# Step 3 — Train all 5 models
python train_model.py

# Step 4 — Launch the app
streamlit run app.py
```


## 📂 What Gets Created Automatically

After running all steps your folder will look like this:

```
loan_guard_ai/
│
├── generate_data.py
├── eda.py
├── train_model.py
├── app.py
├── requirements.txt
│
├── data/                          ← created by generate_data.py & eda.py
│   ├── loan_applications.csv
│   ├── transactions.csv
│   └── cleaned_loan_data.csv
│
├── plots/                         ← created by eda.py
│   ├── loan_status_dist.png
│   ├── correlation_map.png
│   └── cibil_distribution.png
│
└── models/                        ← created by train_model.py
    ├── loan_model.pkl             ← best model
    ├── all_models.pkl             ← all 5 models
    ├── label_encoder.pkl          ← loan status encoder
    ├── encoders.pkl               ← category encoders
    ├── model_columns.pkl          ← feature order
    ├── model_results.json         ← accuracy of all 5 models
    └── best_model_name.txt        ← name of best model
```

---

## 🧠 The 5 Models
1. Random Forest
2. Naive Bayes
3. Logistic Regression
4. Decision Tree
5. Gradient Boosting

---

## 🔑 Loan Status Classes
- ✅ Approved
- ❌ Declined
- 🚨 Fraudulent - Detected
- ⚠️ Fraudulent - Undetected
