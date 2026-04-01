
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AP Anomaly Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("ap_transactions_results.csv",
                     parse_dates=["invoice_date", "due_date", "payment_date"])
    return df

df = load_data()

# ── Sidebar filters ──────────────────────────────────────────
st.sidebar.title("🔍 AP Anomaly Detector")
st.sidebar.markdown("---")

# Date range
min_date = df["invoice_date"].min().date()
max_date = df["invoice_date"].max().date()
date_from, date_to = st.sidebar.date_input(
    "Invoice Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Vendor filter
all_vendors = ["All Vendors"] + sorted(df["vendor"].unique().tolist())
selected_vendor = st.sidebar.selectbox("Vendor", all_vendors)

# Department filter
all_depts = ["All Departments"] + sorted(df["department"].unique().tolist())
selected_dept = st.sidebar.selectbox("Department", all_depts)

# Show only flagged toggle
show_flagged_only = st.sidebar.checkbox("Show flagged transactions only", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Model: Isolation Forest | Features: 7 | Contamination: 5.7%")

# ── Apply filters ─────────────────────────────────────────────
mask = (
    (df["invoice_date"].dt.date >= date_from) &
    (df["invoice_date"].dt.date <= date_to)
)
if selected_vendor != "All Vendors":
    mask &= df["vendor"] == selected_vendor
if selected_dept != "All Departments":
    mask &= df["department"] == selected_dept
if show_flagged_only:
    mask &= df["predicted_anomaly"] == 1

filtered = df[mask].copy()

# ── Title ─────────────────────────────────────────────────────
st.title("🏦 Accounts Payable — Cash Flow Anomaly Detector")
st.markdown("Unsupervised anomaly detection on retail AP transactions using **Isolation Forest**.")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

total        = len(filtered)
flagged      = filtered["predicted_anomaly"].sum()
flagged_amt  = filtered[filtered["predicted_anomaly"] == 1]["invoice_amount"].sum()
avg_delay    = filtered["payment_delay_days"].mean()
partial_pct  = (filtered["payment_ratio"] < 1.0).mean() * 100

k1.metric("Total Transactions", f"{total:,}")
k2.metric("🚨 Flagged",          f"{flagged:,}",
          delta=f"{100*flagged/total:.1f}% of total" if total > 0 else "")
k3.metric("💰 Flagged Amount",   f"SAR {flagged_amt:,.0f}")
k4.metric("⏱ Avg Payment Delay", f"{avg_delay:.1f} days")
k5.metric("⚠️ Partial Payments", f"{partial_pct:.1f}%")

st.markdown("---")

# ── Charts Row 1 ──────────────────────────────────────────────
col1, col2 = st.columns(2)

# Chart 1: Anomaly score over time
with col1:
    st.subheader("Anomaly Score Over Time")
    fig1 = px.scatter(
        filtered,
        x="invoice_date",
        y="anomaly_score",
        color=filtered["predicted_anomaly"].map({0: "Normal", 1: "Flagged"}),
        color_discrete_map={"Normal": "#2196F3", "Flagged": "#F44336"},
        hover_data=["transaction_id", "vendor", "invoice_amount",
                    "payment_delay_days", "anomaly_type"],
        labels={"invoice_date": "Invoice Date",
                "anomaly_score": "Anomaly Score",
                "color": "Status"},
        height=350
    )
    fig1.update_traces(marker=dict(size=7, opacity=0.7))
    st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Invoice amount by vendor (flagged vs normal)
with col2:
    st.subheader("Invoice Amount by Vendor")
    vendor_summary = (
        filtered.groupby(["vendor", filtered["predicted_anomaly"]
                          .map({0: "Normal", 1: "Flagged"})])
        ["invoice_amount"].sum().reset_index()
    )
    vendor_summary.columns = ["vendor", "status", "total_amount"]
    fig2 = px.bar(
        vendor_summary,
        x="vendor", y="total_amount", color="status",
        color_discrete_map={"Normal": "#2196F3", "Flagged": "#F44336"},
        barmode="stack",
        labels={"vendor": "Vendor", "total_amount": "Total Amount (SAR)",
                "status": "Status"},
        height=350
    )
    fig2.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig2, use_container_width=True)

# ── Charts Row 2 ──────────────────────────────────────────────
col3, col4 = st.columns(2)

# Chart 3: Payment delay distribution
with col3:
    st.subheader("Payment Delay Distribution")
    fig3 = px.histogram(
        filtered,
        x="payment_delay_days",
        color=filtered["predicted_anomaly"].map({0: "Normal", 1: "Flagged"}),
        color_discrete_map={"Normal": "#2196F3", "Flagged": "#F44336"},
        nbins=40, barmode="overlay", opacity=0.65,
        labels={"payment_delay_days": "Days Past Due Date",
                "color": "Status"},
        height=320
    )
    fig3.add_vline(x=0, line_dash="dash", line_color="grey",
                   annotation_text="Due date")
    st.plotly_chart(fig3, use_container_width=True)

# Chart 4: Monthly flagged amount
with col4:
    st.subheader("Monthly Flagged Transaction Amount")
    flagged_df = filtered[filtered["predicted_anomaly"] == 1].copy()
    if not flagged_df.empty:
        flagged_df["month"] = flagged_df["invoice_date"].dt.to_period("M").astype(str)
        monthly = flagged_df.groupby("month")["invoice_amount"].sum().reset_index()
        fig4 = px.bar(
            monthly, x="month", y="invoice_amount",
            color_discrete_sequence=["#F44336"],
            labels={"month": "Month", "invoice_amount": "Flagged Amount (SAR)"},
            height=320
        )
        fig4.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No flagged transactions in this date range.")

st.markdown("---")

# ── Flagged Transactions Table ─────────────────────────────────
st.subheader("📋 Flagged Transactions")

display_cols = [
    "transaction_id", "vendor", "department",
    "invoice_date", "invoice_amount", "paid_amount",
    "payment_delay_days", "payment_ratio",
    "anomaly_score", "anomaly_type", "predicted_anomaly"
]

table_df = filtered[filtered["predicted_anomaly"] == 1][display_cols].copy()
table_df = table_df.sort_values("anomaly_score", ascending=False)
table_df["invoice_date"]    = table_df["invoice_date"].dt.date
table_df["invoice_amount"]  = table_df["invoice_amount"].map("SAR {:,.2f}".format)
table_df["paid_amount"]     = table_df["paid_amount"].map("SAR {:,.2f}".format)
table_df["payment_ratio"]   = table_df["payment_ratio"].map("{:.0%}".format)
table_df["anomaly_score"]   = table_df["anomaly_score"].map("{:.3f}".format)
table_df["predicted_anomaly"] = table_df["predicted_anomaly"].map({1: "🚨 Flagged"})

table_df.columns = [
    "ID", "Vendor", "Dept", "Invoice Date",
    "Invoice Amt", "Paid Amt", "Delay (days)",
    "Pay Ratio", "Score", "True Type", "Status"
]

if table_df.empty:
    st.info("No flagged transactions match the current filters.")
else:
    st.dataframe(table_df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(table_df)} flagged transaction(s)")

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with Python · Isolation Forest (scikit-learn) · Streamlit · Plotly | "
    "Portfolio project — Accounts Payable Anomaly Detection"
)
