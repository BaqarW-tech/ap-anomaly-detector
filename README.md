🏦 Accounts Payable Cash Flow Anomaly Detector

An end-to-end unsupervised anomaly detection pipeline for Accounts Payable (AP) transactions.
This project simulates financial transaction data, engineers fraud-risk features, trains a machine learning model to detect anomalies, and presents results through an interactive dashboard.

The system is designed as a portfolio project demonstrating data analytics, machine learning, and financial risk analysis.


📊 Project Overview

Organizations process thousands of invoices and payments. Detecting abnormal transactions manually is difficult.

This project builds a pipeline that:

1. Generates realistic AP transaction data
2. Engineers financial risk indicators
3. Trains an anomaly detection model
4. Displays flagged transactions in a dashboard

The goal is to identify suspicious payment patterns automatically.


⚙️ Pipeline Architecture

Step| Script| Output
Data Generation| "ap_data_generator.py"| "ap_transactions.csv"
Feature Engineering & EDA| "ap_step2_features_eda.py"| "ap_transactions_features.csv"
Model Training| "ap_step3_model.py"| "ap_transactions_results.csv"
Dashboard| "app.py"| Interactive Streamlit dashboard


🚨 Anomaly Types Detected

The pipeline detects multiple financial risk scenarios:

Anomaly Type| Description
High Amount| Invoice far above vendor’s normal range
Late Payment| Payment significantly past due date
Partial Payment| Only a fraction of invoice was paid
Duplicate Vendor| Same vendor and amount within a short window
Zero Amount| Invalid or suspicious invoice value


🔬 Feature Engineering

Several risk indicators are created from raw transactions.

Feature| Purpose
"payment_delay_days"| Detect late payments
"processing_time_days"| Identify operational delays
"payment_ratio"| Capture partial or zero payments
"amount_zscore_vendor"| Compare invoice to vendor norms
"amount_zscore_overall"| Compare invoice to overall distribution
"vendor_monthly_count"| Detect duplicate vendor behavior
"log_amount"| Handle extreme amount distributions


🤖 Machine Learning Model

The system uses Isolation Forest, an unsupervised anomaly detection algorithm.

Model characteristics

- Algorithm: Isolation Forest
- Library: scikit-learn
- Mode: Unsupervised learning
- Estimators: 200 trees
- Contamination: ~5–6% anomalies

Isolation Forest works by isolating rare observations faster than normal observations, making it well suited for financial anomaly detection.


📈 Dashboard Features

The interactive dashboard (built with Streamlit and Plotly) allows users to explore results.

Features include:

- Date range filtering
- Vendor and department filters
- KPI summary cards
- Time-series anomaly visualization
- Vendor payment comparison charts
- Payment delay distributions
- Monthly anomaly counts
- Table of flagged transactions


🚀 Quick Start

1. Clone the repository

git clone https://github.com/yourusername/ap-anomaly-detector.git
cd ap-anomaly-detector

2. Install dependencies

pip install -r requirements.txt

3. Run the data pipeline

python ap_data_generator.py
python ap_step2_features_eda.py
python ap_step3_model.py

4. Launch the dashboard

streamlit run app.py

The dashboard will open in your browser.


🧰 Tech Stack

Programming

- Python

Data Processing

- pandas
- NumPy

Machine Learning

- scikit-learn (Isolation Forest)

Visualization

- Plotly
- Matplotlib

Dashboard

- Streamlit


📁 Project Structure

ap-anomaly-detector/
│
├── ap_data_generator.py
├── ap_step2_features_eda.py
├── ap_step3_model.py
├── app.py
├── requirements.txt
├── README.md
│
└── outputs/
    ├── ap_transactions.csv
    ├── ap_transactions_features.csv
    ├── ap_transactions_results.csv


📌 Use Cases

This system demonstrates techniques useful for:

- Financial fraud detection
- Accounts payable monitoring
- Audit analytics
- Risk management dashboards
- Data analytics portfolios


📚 Learning Objectives

This project demonstrates:

- Feature engineering for financial data
- Unsupervised anomaly detection
- Model evaluation techniques
- Interactive data dashboards
- End-to-end data pipeline design


📜 License

This project is open-source and available under the MIT License.


👤 Author

Developed as a portfolio project in financial data analytics and machine learning.

GitHub: https://github.com/BaqarW-tech
