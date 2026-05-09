# dashboard.py
"""
Streamlit monitoring UI for the weather arbitrage bot.

Run:
    streamlit run dashboard.py --server.port 8501

Reads from bot/data/signals.db. Auto-refreshes every 60s; "Refresh Data"
button forces a cache clear. WAL mode + 30s timeout means this can run
concurrently with main.py without lock contention.
"""

import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DB_PATH


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Weather Arb Dashboard",
    page_icon="🌦️",
    layout="wide",
)
st.title("Weather Arbitrage Bot Dashboard")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_signals() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        df = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC", conn)
    finally:
        conn.close()
    return df


@st.cache_data(ttl=60)
def load_positions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        df = pd.read_sql("SELECT * FROM positions ORDER BY entry_time DESC", conn)
    finally:
        conn.close()
    return df


# Try to load; if the schema isn't initialized yet, give a friendly nudge
try:
    signals_df = load_signals()
    positions_df = load_positions()
except Exception as e:
    st.warning(
        f"Could not read database at `{DB_PATH}` — has the bot run yet?\n\n"
        f"Try: `cd bot && python main.py --dry-run`\n\nError: `{e}`"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_pnl = positions_df["pnl"].sum() if not positions_df.empty and "pnl" in positions_df else 0.0
    st.metric("Cumulative P&L", f"${total_pnl:.2f}")

with col2:
    open_count = (
        len(positions_df[positions_df["status"] == "open"])
        if not positions_df.empty and "status" in positions_df else 0
    )
    st.metric("Open Positions", open_count)

with col3:
    total_signals = len(signals_df)
    st.metric("Total Signals", total_signals)

with col4:
    if not signals_df.empty and "ev" in signals_df and signals_df["ev"].notna().any():
        avg_ev = signals_df["ev"].mean()
        st.metric("Avg Signal EV", f"{avg_ev:.3f}")
    else:
        st.metric("Avg Signal EV", "N/A")

# Bot mode banner — quick at-a-glance status
from config import PAPER_TRADE  # late import after st.set_page_config
mode_emoji = "🧪" if PAPER_TRADE else "💰"
mode_label = "PAPER" if PAPER_TRADE else "LIVE"
st.caption(f"{mode_emoji}  Mode: **{mode_label}**  |  Database: `{DB_PATH}`")

st.divider()


# ---------------------------------------------------------------------------
# Cumulative P&L
# ---------------------------------------------------------------------------

st.subheader("Cumulative P&L Over Time")

if (
    not positions_df.empty
    and "pnl" in positions_df
    and "status" in positions_df
    and "exit_time" in positions_df
):
    closed = positions_df[positions_df["status"] == "closed"].copy()
    if not closed.empty:
        closed = closed.sort_values("exit_time")
        closed["cumulative_pnl"] = closed["pnl"].cumsum()
        fig = px.line(
            closed, x="exit_time", y="cumulative_pnl",
            labels={"exit_time": "Date", "cumulative_pnl": "Cumulative P&L ($)"},
        )
        fig.update_traces(line_color="#00CC96")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No closed positions yet — cumulative P&L chart appears once trades resolve.")
else:
    st.info("No position data yet.")


# ---------------------------------------------------------------------------
# Model calibration
# ---------------------------------------------------------------------------

st.subheader("Model Calibration")
st.caption(
    "Bars show the actual outcome rate for each model-probability bucket. "
    "Perfect calibration = bars track the diagonal red line."
)

if (
    not signals_df.empty
    and "model_p" in signals_df
    and "outcome" in signals_df
    and signals_df["outcome"].notna().any()
):
    executed = signals_df[(signals_df["executed"] == 1) & (signals_df["outcome"].notna())].copy()
    # Coerce outcome to numeric 0/1 (it's stored as TEXT 'YES'/'NO' or '1'/'0')
    executed["outcome_num"] = executed["outcome"].apply(
        lambda v: 1 if str(v).upper() in ("YES", "1") else 0
    )
    if len(executed) >= 5:
        executed["p_bucket"] = pd.cut(executed["model_p"], bins=10, include_lowest=True)
        cal = executed.groupby("p_bucket", observed=True).agg(
            avg_model_p=("model_p", "mean"),
            avg_outcome=("outcome_num", "mean"),
            count=("outcome_num", "count"),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cal["avg_model_p"], y=cal["avg_outcome"],
            name="Actual Frequency", marker_color="steelblue",
            customdata=cal["count"],
            hovertemplate="model_p=%{x:.2f}<br>outcome=%{y:.2f}<br>n=%{customdata}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Perfect Calibration", line=dict(color="red", dash="dash"),
        ))
        fig.update_layout(xaxis_title="Model Probability", yaxis_title="Actual Outcome Rate")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info(f"Need at least 5 resolved trades for calibration (have {len(executed)}).")
else:
    st.info("Calibration chart appears once we have resolved trades with outcomes recorded.")


# ---------------------------------------------------------------------------
# Recent signals
# ---------------------------------------------------------------------------

st.subheader("Recent Signals")
if not signals_df.empty:
    cols = ["timestamp", "question", "recommended_side", "model_p", "market_p",
            "ev", "kelly_size", "executed"]
    avail = [c for c in cols if c in signals_df.columns]
    st.dataframe(
        signals_df[avail].head(20),
        width='stretch',
        column_config={
            "ev":         st.column_config.NumberColumn("EV",        format="%.3f"),
            "model_p":    st.column_config.NumberColumn("Model P",   format="%.3f"),
            "market_p":   st.column_config.NumberColumn("Market P",  format="%.3f"),
            "kelly_size": st.column_config.NumberColumn("Size $",    format="$%.2f"),
        },
    )
else:
    st.info("No signals yet. Bot will populate these as weather markets become available.")


# ---------------------------------------------------------------------------
# Open positions
# ---------------------------------------------------------------------------

st.subheader("Open Positions")
if not positions_df.empty:
    open_pos = positions_df[positions_df["status"] == "open"]
    if not open_pos.empty:
        st.dataframe(open_pos, width='stretch')
    else:
        st.info("No open positions.")
else:
    st.info("No position data.")


# ---------------------------------------------------------------------------
# Refresh + footer
# ---------------------------------------------------------------------------

if st.button("🔄  Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.caption("Data auto-refreshes every 60 seconds. Click 'Refresh Data' to force a reload.")
