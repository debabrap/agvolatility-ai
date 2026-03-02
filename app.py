import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="AgVolatility AI", layout="wide")

FEATURES_PATH = os.path.join("features", "all_commodities_volatility_features.csv")

@st.cache_data
def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Commodity"] = df["Commodity"].astype(str).str.upper()
    df = df.sort_values(["Commodity", "Date"])
    return df

def hedge_reco(vol30: float) -> str:
    if np.isnan(vol30):
        return "N/A"
    if vol30 < 0.20:
        return "20–30% (Low volatility)"
    if vol30 < 0.35:
        return "30–50% (Moderate volatility)"
    if vol30 < 0.50:
        return "50–70% (Elevated volatility)"
    return "70%+ (High stress regime)"

def money(x: float) -> str:
    return "${:,.0f}".format(x)

st.title("AgVolatility AI — Commodity Volatility & Margin Exposure")

if not os.path.exists(FEATURES_PATH):
    st.error(f"Could not find {FEATURES_PATH}. Make sure you're running from C:\\AgVolatility and you generated features first.")
    st.stop()

df = load_features(FEATURES_PATH)

# ---- Build KPI table ----
df["Return_7d"] = df.groupby("Commodity")["Price"].pct_change(7)
df["Return_30d"] = df.groupby("Commodity")["Price"].pct_change(30)

latest = df.groupby("Commodity").tail(1).copy()

highest_risk = latest.sort_values("Vol_30d_annual", ascending=False).iloc[0]

st.divider()
st.subheader("Enterprise Risk Status")

col1, col2, col3 = st.columns(3)

col1.metric("Highest Risk Commodity", highest_risk["Commodity"])
col2.metric("30D Vol", f"{highest_risk['Vol_30d_annual']:.3f}")
col3.metric("Risk Level", highest_risk["Risk_30d"])

if highest_risk["Vol_30d_annual"] > 0.25:
    st.error("Action Required: Trigger Procurement Risk Review")
else:
    st.success("Risk Within Acceptable Range")

import numpy as np

vol30 = highest_risk["Vol_30d_annual"]
price = highest_risk["Price"]

# 30-day volatility approximation
monthly_vol = vol30 / np.sqrt(12)

upper = price * (1 + monthly_vol)
lower = price * (1 - monthly_vol)

st.markdown("### Expected 30-Day Price Range (1σ)")
st.info(f"{highest_risk['Commodity']}: {lower:.2f}  —  {upper:.2f}")

kpi = latest[[
    "Commodity", "Date", "Price",
    "Vol_30d_annual", "Risk_30d",
    "Vol_90d_annual", "Risk_90d",
    "Return_7d", "Return_30d"
]].copy()

kpi["Return_7d_%"] = (kpi["Return_7d"] * 100).round(2)
kpi["Return_30d_%"] = (kpi["Return_30d"] * 100).round(2)
kpi["Vol_30d_annual"] = kpi["Vol_30d_annual"].round(3)
kpi["Vol_90d_annual"] = kpi["Vol_90d_annual"].round(3)
kpi["Price"] = kpi["Price"].round(2)
kpi = kpi.drop(columns=["Return_7d", "Return_30d"]).sort_values("Commodity")

# ---- Layout ----
left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.subheader("Executive KPI Snapshot (Latest)")
    st.dataframe(
        kpi,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Volatility is rolling std-dev of daily returns (30/90 days), annualized by √252. "
        "Risk buckets are simple thresholds; tune later for your business."
    )

with right:
    st.subheader("Commodity Selection")
    commodities = sorted(df["Commodity"].unique().tolist())
    selected = st.selectbox("Select commodity", commodities, index=min(0, len(commodities)-1))

    row = latest[latest["Commodity"] == selected].iloc[0]
    st.metric("Latest Price", f"{row['Price']:.2f}", help="Latest available close/adj close from the dataset")
    st.metric("30D Vol (annualized)", f"{row['Vol_30d_annual']:.3f}", help="Rolling 30-day std of daily returns × √252")
    st.metric("30D Risk", f"{row['Risk_30d']}", help="Risk bucket based on 30D annualized volatility")
    st.metric("Hedge Recommendation", hedge_reco(float(row["Vol_30d_annual"])))

st.divider()

# ---- Charts ----
st.subheader("Trends (Last 12 Months)")
cutoff = df["Date"].max() - pd.Timedelta(days=365)
d = df[(df["Commodity"] == selected) & (df["Date"] >= cutoff)].copy()

c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("**Price**")
    fig = plt.figure()
    plt.plot(d["Date"], d["Price"])
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with c2:
    st.markdown("**30D Annualized Volatility**")
    fig2 = plt.figure()
    plt.plot(d["Date"], d["Vol_30d_annual"])
    plt.xlabel("Date")
    plt.ylabel("Vol (annualized)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()

# ---- Margin Exposure Simulator ----
st.subheader("Margin Exposure Simulator")

sim_left, sim_right = st.columns([1.0, 1.0], gap="large")

with sim_left:
    annual_spend = st.number_input("Annual spend ($)", min_value=0.0, value=50_000_000.0, step=1_000_000.0, format="%.0f")
    scenario_move_pct = st.number_input("Scenario price move (%)", value=10.0, step=1.0, format="%.2f")
    hedge_coverage_pct = st.number_input("Hedge coverage (%)", min_value=0.0, max_value=100.0, value=30.0, step=5.0, format="%.2f")
    passthrough_pct = st.number_input("Pass-through (%)", min_value=0.0, max_value=100.0, value=0.0, step=5.0, format="%.2f")

move = scenario_move_pct / 100.0
hedge = hedge_coverage_pct / 100.0
passthrough = passthrough_pct / 100.0

gross_impact = annual_spend * move
net_after_hedge = gross_impact * (1 - hedge)
net_after_passthrough = net_after_hedge * (1 - passthrough)

with sim_right:
    st.markdown("**Results**")
    st.write(f"**Gross cost impact:** {money(gross_impact)}")
    st.write(f"**Net exposure after hedge:** {money(net_after_hedge)}")
    st.write(f"**Net exposure after pass-through:** {money(net_after_passthrough)}")

    vol30 = float(row["Vol_30d_annual"])
    if vol30 >= 0.50:
        guidance = "Extreme volatility regime — tighten hedging bands and re-forecast more frequently."
    elif vol30 >= 0.35:
        guidance = "Elevated volatility — consider increasing hedge ratio and shortening planning cycles."
    elif vol30 >= 0.20:
        guidance = "Moderate volatility — monitor weekly; scenario plan procurement and pricing."
    else:
        guidance = "Low volatility — standard planning cadence likely sufficient."

    st.markdown("**Risk guidance**")
    st.info(guidance)

    st.markdown("**Suggested hedge band**")
    st.success(hedge_reco(vol30))

st.caption("Note: This is a scenario stress test (not a price-direction forecast). The purpose is decision support for procurement/hedging.")

st.divider()
st.subheader("Enterprise Portfolio Exposure")

portfolio_left, portfolio_right = st.columns(2)

with portfolio_left:
    wheat_spend = st.number_input("Annual Wheat Spend ($)", value=50_000_000.0, step=1_000_000.0)
    corn_spend = st.number_input("Annual Corn Spend ($)", value=30_000_000.0, step=1_000_000.0)
    soy_spend = st.number_input("Annual Soybean Spend ($)", value=20_000_000.0, step=1_000_000.0)
    scenario_move = st.number_input("Scenario price move (%) for all commodities", value=10.0)

move = scenario_move / 100.0

gross_portfolio = (
    wheat_spend * move +
    corn_spend * move +
    soy_spend * move
)

with portfolio_right:
    st.markdown("### Portfolio Impact")
    st.write(f"Gross Portfolio Cost Impact: {money(gross_portfolio)}")

    # Identify highest current risk commodity
    highest_risk = latest.sort_values("Vol_30d_annual", ascending=False).iloc[0]
    st.write(f"Highest Volatility Commodity: {highest_risk['Commodity']}")
    st.write(f"Current 30D Vol: {highest_risk['Vol_30d_annual']:.3f}")

    if highest_risk["Vol_30d_annual"] > 0.25:
        st.error("Action: Trigger Procurement Risk Review")
    else:
        st.success("Portfolio Risk Within Acceptable Range")