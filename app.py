"""
Gold Trade Planner — Streamlit Web App
AutoTheta + AutoETS ensemble, multi-timeframe alignment, full risk management.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsforecast import StatsForecast
from statsforecast.models import AutoTheta, AutoETS
from datetime import datetime, timezone

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Trade Planner",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
TICKER     = "GC=F"
TRAIN_BARS = 200

TIMEFRAMES = {
    "15 min": {"interval": "15m", "period": "60d", "season": 26, "freq": "15min"},
    "1 hour": {"interval": "60m", "period": "60d", "season": 24, "freq": "h"},
    "4 hour": {"interval": "60m", "period": "60d", "season": 6,  "freq": "h",  "resample": "4h"},
    "Daily":  {"interval": "1d",  "period": "2y",  "season": 5,  "freq": "D"},
}


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_atr(df, period=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series, period=14):
    d    = series.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_adx(df, period=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    pdm = hi.diff().clip(lower=0)
    ndm = (-lo.diff()).clip(lower=0)
    pdm[pdm < ndm] = 0
    ndm[ndm < pdm] = 0
    tr    = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi   = 100 * pdm.ewm(alpha=1 / period, adjust=False).mean() / atr14
    ndi   = 100 * ndm.ewm(alpha=1 / period, adjust=False).mean() / atr14
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx   = dx.ewm(alpha=1 / period, adjust=False).mean()
    return float(adx.iloc[-1]), float(pdi.iloc[-1]), float(ndi.iloc[-1])


# ── Data & forecast ───────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(interval: str, period: str, resample: str | None = None) -> pd.DataFrame:
    df = yf.download(TICKER, interval=interval, period=period, progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    if resample:
        df = df.resample(resample).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
    return df


def run_forecast(df: pd.DataFrame, season: int, freq: str, h: int = 4):
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    series  = log_ret.iloc[-TRAIN_BARS:].values.astype(float)
    n       = len(series)
    sf_df   = pd.DataFrame({
        "unique_id": ["gold"] * n,
        "ds": pd.date_range("2000-01-01", periods=n, freq=freq),
        "y": series,
    })
    preds = StatsForecast(
        models=[AutoTheta(season_length=season), AutoETS(season_length=season)],
        freq=freq, n_jobs=1,
    ).forecast(df=sf_df, h=h)
    theta = preds["AutoTheta"].values
    ets   = preds["AutoETS"].values
    return theta, ets, (theta + ets) / 2


# ── Signal logic ──────────────────────────────────────────────────────────────

def compute_confidence(direction, rsi, adx_val, pdi, ndi, ema20, cur_px, tf_dirs):
    score = 50
    if adx_val > 25:    score += 10
    elif adx_val > 20:  score += 5
    if direction == "BUY"  and pdi > ndi: score += 10
    if direction == "SELL" and ndi > pdi: score += 10
    if direction == "BUY"  and 40 < rsi < 65: score += 8
    if direction == "SELL" and 35 < rsi < 60: score += 8
    if direction == "SELL" and cur_px < ema20: score += 7
    if direction == "BUY"  and cur_px > ema20: score += 7
    aligned = sum(1 for d in tf_dirs.values() if d == direction)
    score  += int((aligned / max(len(tf_dirs), 1)) * 15)
    return min(max(score, 0), 100)


def compute_entry(direction, cur_px, ema20):
    if direction == "SELL":
        if cur_px < ema20:
            return round(ema20, 2), "LIMIT — Sell at EMA20 (resistance)"
        return round(cur_px, 2), "MARKET — Sell at current close"
    if cur_px > ema20:
        return round(ema20, 2), "LIMIT — Buy at EMA20 (support)"
    return round(cur_px, 2), "MARKET — Buy at current close"


def compute_sl_tp(df, direction, entry, atr14, swing_bars, atr_mult, rr_ratio):
    s_high = float(df["high"].tail(swing_bars).max())
    s_low  = float(df["low"].tail(swing_bars).min())
    atr_sl = atr14 * atr_mult
    if direction == "SELL":
        sl      = round(max(entry + atr_sl, s_high + atr14 * 0.3), 2)
        sl_dist = sl - entry
        tp      = round(max(entry - sl_dist * rr_ratio, s_low - atr14 * 0.2), 2)
    else:
        sl      = round(min(entry - atr_sl, s_low - atr14 * 0.3), 2)
        sl_dist = entry - sl
        tp      = round(min(entry + sl_dist * rr_ratio, s_high + atr14 * 0.2), 2)
    return sl, tp, abs(sl_dist)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    entry_tf   = st.selectbox("Entry Timeframe", list(TIMEFRAMES.keys()))
    st.divider()
    st.subheader("Risk Management")
    account    = st.number_input("Account Balance (USD)", 100, 1_000_000, 1_000, 100)
    risk_pct   = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5) / 100
    rr_ratio   = st.slider("R:R Ratio", 1.0, 5.0, 2.0, 0.5)
    atr_mult   = st.slider("ATR Stop Loss Multiplier", 1.0, 3.0, 1.5, 0.25)
    swing_bars = st.slider("Swing Bars Lookback", 10, 50, 20, 5)
    st.divider()
    st.caption("Data cached 5 min. Click Generate to refresh.")


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Gold (XAUUSD) Trade Planner")
st.caption(
    f"AutoTheta + AutoETS Ensemble  |  Live via yfinance  |  "
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)

generate = st.button("Generate Signal", type="primary", use_container_width=True)

if not generate:
    st.info("Configure settings in the sidebar, then click **Generate Signal**.")
    st.markdown("""
**How it works:**
1. Fetches live Gold data across 4 timeframes (15m, 1h, 4h, Daily)
2. Runs AutoTheta + AutoETS ensemble on each timeframe
3. Scores confidence from 5 factors: TF alignment, ADX trend, RSI, EMA position, DI cross
4. Calculates Entry (limit or market), Stop Loss (ATR + swing level), Take Profit
5. Sizes position using fixed-fractional risk management
""")
    st.stop()

# ── Run analysis ──────────────────────────────────────────────────────────────

with st.spinner("Fetching live Gold data and running ensemble models..."):

    cfg = TIMEFRAMES[entry_tf]
    df  = fetch_data(cfg["interval"], cfg["period"], cfg.get("resample"))

    cur_px  = float(df["close"].iloc[-1])
    atr14   = float(compute_atr(df, 14).iloc[-1])
    ema20_v = float(compute_ema(df["close"], 20).iloc[-1])
    ema50_v = float(compute_ema(df["close"], 50).iloc[-1])
    rsi14   = float(compute_rsi(df["close"], 14).iloc[-1])
    adx_val, pdi, ndi = compute_adx(df, 14)

    theta, ets, ens = run_forecast(df, cfg["season"], cfg["freq"], h=4)
    direction = "SELL" if ens[0] < 0 else "BUY"
    arrow     = "▼" if direction == "SELL" else "▲"

    # Multi-TF sweep
    tf_dirs    = {}
    tf_returns = {}
    for tf_name, tcfg in TIMEFRAMES.items():
        try:
            df_tf = fetch_data(tcfg["interval"], tcfg["period"], tcfg.get("resample"))
            _, _, e = run_forecast(df_tf, tcfg["season"], tcfg["freq"], h=1)
            tf_dirs[tf_name]    = "SELL" if e[0] < 0 else "BUY"
            tf_returns[tf_name] = float(e[0])
        except Exception:
            tf_dirs[tf_name]    = "?"
            tf_returns[tf_name] = 0.0

    confidence = compute_confidence(direction, rsi14, adx_val, pdi, ndi, ema20_v, cur_px, tf_dirs)
    entry, entry_type = compute_entry(direction, cur_px, ema20_v)
    sl, tp, sl_dist   = compute_sl_tp(df, direction, entry, atr14, swing_bars, atr_mult, rr_ratio)
    risk_usd   = account * risk_pct
    pos_oz     = round(risk_usd / sl_dist, 4) if sl_dist > 0 else 0
    reward_usd = round(pos_oz * sl_dist * rr_ratio, 2)
    rr_actual  = abs(tp - entry) / sl_dist if sl_dist > 0 else 0


# ── Layout ────────────────────────────────────────────────────────────────────

# Signal header
sig_color = "green" if direction == "BUY" else "red"
col_sig, col_conf, col_px, col_tf = st.columns([3, 1, 1, 1])
with col_sig:
    st.markdown(f"## {arrow} {direction} &nbsp; XAUUSD [{entry_tf}]")
    st.markdown(f"*{entry_type}*")
with col_conf:
    st.metric("Confidence", f"{confidence}%")
with col_px:
    st.metric("Price", f"${cur_px:,.2f}")
with col_tf:
    aligned = sum(1 for d in tf_dirs.values() if d == direction)
    st.metric("TF Aligned", f"{aligned}/{len(tf_dirs)}")

st.divider()

# Market context row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ATR(14)",  f"${atr14:.2f}")
c2.metric("RSI(14)",  f"{rsi14:.1f}")
c3.metric("ADX(14)",  f"{adx_val:.1f}", delta="Trending" if adx_val > 25 else "Ranging")
c4.metric("EMA20",    f"${ema20_v:,.2f}", delta="Above" if cur_px > ema20_v else "Below")
c5.metric("EMA50",    f"${ema50_v:,.2f}", delta="Above" if cur_px > ema50_v else "Below")

st.divider()

# Multi-TF alignment
st.subheader("Multi-Timeframe Alignment")
tf_cols = st.columns(len(TIMEFRAMES))
for i, (tf_name, d) in enumerate(tf_dirs.items()):
    with tf_cols[i]:
        icon  = "▲" if d == "BUY" else ("▼" if d == "SELL" else "?")
        match = d == direction
        st.metric(
            tf_name, f"{icon} {d}",
            delta="Aligned" if match else "Conflict",
            delta_color="normal" if match else "inverse",
        )

st.divider()

# Forecast table + Trade plan side by side
left, right = st.columns(2)

with left:
    st.subheader("Ensemble Forecast")
    rows = []
    for h in [1, 2, 4]:
        cum_ens = sum(ens[:h])
        rows.append({
            "Horizon":      f"+{h} bar{'s' if h > 1 else ''}",
            "AutoTheta":    f"{sum(theta[:h])*100:+.4f}%",
            "AutoETS":      f"{sum(ets[:h])*100:+.4f}%",
            "Ensemble":     f"{cum_ens*100:+.4f}%",
            "Price Target": f"${cur_px * np.exp(cum_ens):,.2f}",
            "Direction":    "▲ UP" if cum_ens > 0 else "▼ DOWN",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with right:
    st.subheader("Trade Plan")
    plan_df = pd.DataFrame([
        {"Field": "Direction",     "Value": f"{arrow} {direction}"},
        {"Field": "Entry",         "Value": f"${entry:,.3f}"},
        {"Field": "Stop Loss",     "Value": f"${sl:,.3f}  ({sl_dist:.2f} pts = {atr_mult}x ATR)"},
        {"Field": "Take Profit",   "Value": f"${tp:,.3f}  ({abs(tp-entry):.2f} pts)"},
        {"Field": "R:R",           "Value": f"1:{rr_actual:.2f}"},
        {"Field": "Account",       "Value": f"${account:,.0f}"},
        {"Field": "Risk",          "Value": f"{risk_pct:.1%} = ${risk_usd:.2f}"},
        {"Field": "Position Size", "Value": f"{pos_oz:.4f} oz"},
        {"Field": "Max Loss",      "Value": f"${risk_usd:.2f}"},
        {"Field": "Max Gain",      "Value": f"${reward_usd:.2f}"},
    ])
    st.dataframe(plan_df, use_container_width=True, hide_index=True)

# Confidence alert
st.divider()
if confidence >= 70:
    st.success(f"Strong signal — confidence {confidence}%. All key factors aligned.")
elif confidence >= 55:
    st.warning(f"Moderate confidence ({confidence}%). Consider reducing position size by 25-50%.")
else:
    st.error(f"Weak signal — confidence {confidence}%. Timeframes conflicting. Best to skip this trade.")

if direction == "SELL" and rsi14 < 35:
    st.warning("RSI oversold — counter-trend SELL. Higher risk.")
elif direction == "BUY" and rsi14 > 65:
    st.warning("RSI overbought — counter-trend BUY. Higher risk.")

st.divider()

# ── Chart ─────────────────────────────────────────────────────────────────────
st.subheader("Price Chart")

chart_df  = df.tail(120)
ema20_s   = compute_ema(df["close"], 20).tail(120)
ema50_s   = compute_ema(df["close"], 50).tail(120)
rsi_s     = compute_rsi(df["close"], 14).tail(120)
vol_colors = ["green" if c >= o else "red"
              for c, o in zip(chart_df["close"], chart_df["open"])]

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.60, 0.22, 0.18],
    vertical_spacing=0.03,
    subplot_titles=["XAUUSD Price", "RSI(14)", "Volume"],
)

# Candlesticks
fig.add_trace(go.Candlestick(
    x=chart_df.index,
    open=chart_df["open"], high=chart_df["high"],
    low=chart_df["low"],   close=chart_df["close"],
    name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
), row=1, col=1)

fig.add_trace(go.Scatter(x=chart_df.index, y=ema20_s, name="EMA20",
    line=dict(color="orange", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=ema50_s, name="EMA50",
    line=dict(color="#5C9BD6", width=1.5)), row=1, col=1)

# Signal levels
for level, color, label in [
    (entry, "white",  f"Entry  ${entry:,.2f}"),
    (sl,    "#ef5350", f"SL     ${sl:,.2f}"),
    (tp,    "#26a69a", f"TP     ${tp:,.2f}"),
]:
    fig.add_shape(type="line", x0=chart_df.index[0], x1=chart_df.index[-1],
                  y0=level, y1=level, line=dict(color=color, dash="dash", width=1.5), row=1, col=1)
    fig.add_annotation(x=chart_df.index[-1], y=level, text=f" {label}",
                       xanchor="left", showarrow=False, font=dict(color=color, size=11), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=chart_df.index, y=rsi_s, name="RSI",
    line=dict(color="#CE93D8", width=1.5)), row=2, col=1)
for lvl, clr in [(70, "#ef5350"), (30, "#26a69a")]:
    fig.add_shape(type="line", x0=chart_df.index[0], x1=chart_df.index[-1],
                  y0=lvl, y1=lvl, line=dict(color=clr, dash="dot", width=1), row=2, col=1)

# Volume
fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["volume"],
    name="Volume", marker_color=vol_colors, opacity=0.7), row=3, col=1)

fig.update_layout(
    template="plotly_dark", height=720,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=0, r=120, t=30, b=0),
)
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Last bar: {df.index[-1].strftime('%Y-%m-%d %H:%M UTC')}  |  "
    f"Training: {TRAIN_BARS} bars  |  Models: AutoTheta + AutoETS  |  "
    f"Data: yfinance (GC=F)"
)
