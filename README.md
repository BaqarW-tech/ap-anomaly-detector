# 🏦 Accounts Payable / Cash Flow Anomaly Detector

An unsupervised machine learning project that detects unusual transactions,
delayed supplier payments, and cash flow irregularities in retail accounts
payable data.

**Live Demo →** https://share.streamlit.io  *(update this link after deploying)*

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| **Goal** | Flag anomalous AP transactions before they impact cash flow |
| **Model** | Isolation Forest (unsupervised, no labelled training data needed) |
| **Stack** | Python · pandas · scikit-learn · Streamlit · Plotly |
| **Data** | Synthetic retail AP dataset (530 transactions, 10 vendors, 18 months) |

---

## 🔍 Anomaly Types Detected

| Type | Detection Rate | Signal Used |
|---|---|---|
| High Amount | 83% | amount z-score vs vendor & global norm |
| Late Payment | 100% | payment_delay_days |
| Partial Payment | 83% | payment_ratio < 1.0 |
| Zero Amount | 100% | log_amount = 0 |
| Duplicate Vendor | 0% | vendor_monthly_count (weak signal — see note) |

> **Note on Duplicate Vendor:** Isolation Forest relies on feature isolation,
> not exact matching. Detecting true duplicates requires a rule-based layer
> (e.g. same vendor + same amount ± 3 days). This hybrid approach is a
> planned enhancement.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Overall Accuracy | 97% |
| ROC-AUC | 0.916 |
| Precision (anomaly) | 71% |
| Recall (anomaly) | 73% |
| Contamination param | 5.7% |

---

## 🗂 Project Structure

```
ap-anomaly-detector/
│
├── app.py                          # Streamlit dashboard
├── ap_data_generator.py            # Synthetic dataset generator (Step 1)
├── ap_step2_features_eda.py        # Feature engineering + EDA (Step 2)
├── ap_step3_model.py               # Isolation Forest model (Step 3)
│
├── ap_transactions.csv             # Raw synthetic dataset
├── ap_transactions_features.csv    # Dataset with engineered features
├── ap_transactions_results.csv     # Dataset with model predictions
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/BaqarW-tech/ap-anomaly-detector.git
cd ap-anomaly-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data and run the pipeline
python ap_data_generator.py
python ap_step2_features_eda.py
python ap_step3_model.py

# 4. Launch the dashboard
streamlit run app.py
```

---

## 💡 Business Impact

- **Fraud prevention:** Flags inflated invoices and zero-amount entries
  before payment approval
- **Working capital:** Identifies late payments that signal cash flow risk
- **Audit trail:** Interactive dashboard gives finance teams a filterable
  view of all suspicious transactions by vendor, department, and date

---

## 🛠 Tech Stack

- **Python 3.11** — core language
- **pandas / numpy** — data manipulation and feature engineering
- **scikit-learn** — Isolation Forest anomaly detection
- **Streamlit** — interactive web dashboard
- **Plotly** — interactive charts

---

## 👤 Author

**Baqar W.** — Data Analytics Portfolio  
MA Economics | Accounting Background  
Targeting: Data Analytics · Financial Analysis · Vision 2030-aligned roles  
GitHub: [github.com/BaqarW-tech](https://github.com/BaqarW-tech)
