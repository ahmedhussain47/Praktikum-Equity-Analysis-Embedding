"""
Gold 15-Minute Intraday Trade Planner
AutoTheta + AutoETS forecast → proper Entry / SL / TP / Position Size
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsforecast import StatsForecast
from statsforecast.models import AutoTheta, AutoETS
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

# ── Settings ───────────────────────────────────────────────────────────────
TICKER       = "GC=F"
INTERVAL     = "15m"
PERIOD       = "60d"
TRAIN_BARS   = 200
HORIZONS     = [1, 2, 4]

# Risk management
ACCOUNT_USD  = 1000.0   # your account size
RISK_PCT     = 0.01     # 1% risk per trade
ATR_SL_MULT  = 1.5      # SL = entry ± ATR × this
RR_RATIO     = 2.0      # TP = entry ∓ SL_distance × this
SWING_BARS   = 20       # bars to look back for swing high/low SL anchor


# ── Data ───────────────────────────────────────────────────────────────────

def fetch_gold():
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    return df


# ── Indicators ─────────────────────────────────────────────────────────────

def atr(df, period=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat([hi - lo,
                    (hi - cl.shift()).abs(),
                    (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, adjust=False).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    d = series.diff()
    gain = d.clip(lower=0).ewm(alpha=1.0/period, adjust=False).mean()
    loss = (-d).clip(lower=0).ewm(alpha=1.0/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def swing_high(df, n=20):
    return float(df["high"].tail(n).max())


def swing_low(df, n=20):
    return float(df["low"].tail(n).min())


# ── Forecast ───────────────────────────────────────────────────────────────

def run_forecast(df, h):
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    series  = log_ret.iloc[-TRAIN_BARS:].values.astype(float)
    n       = len(series)

    sf_df = pd.DataFrame({
        "unique_id": ["gold"] * n,
        "ds": pd.date_range("2000-01-01", periods=n, freq="15min"),
        "y": series,
    })
    preds = StatsForecast(
        models=[AutoTheta(season_length=26), AutoETS(season_length=26)],
        freq="15min", n_jobs=1
    ).forecast(df=sf_df, h=h)

    theta = preds["AutoTheta"].values
    ets   = preds["AutoETS"].values
    ens   = (theta + ets) / 2
    return theta, ets, ens


# ── Entry logic ────────────────────────────────────────────────────────────

def compute_entry(df, direction, cur_price, atr14, ema20):
    """
    Refined entry price:
    - SELL: enter at EMA20 if price is below EMA20 (EMA acts as resistance),
            otherwise enter at current close (market).
    - BUY:  enter at EMA20 if price is above EMA20 (EMA acts as support),
            otherwise enter at current close.
    Entry type is returned so the user knows whether to place a limit or market.
    """
    if direction == "SELL":
        if cur_price < ema20:
            # Price already below EMA20 — EMA20 is overhead resistance
            # Ideal: limit sell at EMA20 for a better short entry
            entry      = round(ema20, 2)
            entry_type = "LIMIT (sell at EMA20 resistance)"
        else:
            # Price above EMA20 — market sell now
            entry      = round(cur_price, 2)
            entry_type = "MARKET (sell at close)"
    else:
        if cur_price > ema20:
            entry      = round(ema20, 2)
            entry_type = "LIMIT (buy at EMA20 support)"
        else:
            entry      = round(cur_price, 2)
            entry_type = "MARKET (buy at close)"
    return entry, entry_type


def compute_sl_tp(df, direction, entry, atr14):
    """
    SL: tighter of (swing extreme) and (entry ± ATR×mult).
    TP: entry ∓ SL_distance × RR_RATIO.
    """
    atr_sl = atr14 * ATR_SL_MULT

    if direction == "SELL":
        swing_sl  = swing_high(df, SWING_BARS) + atr14 * 0.3   # just above swing high
        sl        = round(max(entry + atr_sl, swing_sl), 2)     # use wider for safety
        sl_dist   = sl - entry
        tp        = round(entry - sl_dist * RR_RATIO, 2)
        swing_tp  = swing_low(df, SWING_BARS)
        tp        = round(max(tp, swing_tp - atr14 * 0.2), 2)   # don't overshoot swing low
    else:
        swing_sl  = swing_low(df, SWING_BARS) - atr14 * 0.3
        sl        = round(min(entry - atr_sl, swing_sl), 2)
        sl_dist   = entry - sl
        tp        = round(entry + sl_dist * RR_RATIO, 2)
        swing_tp  = swing_high(df, SWING_BARS)
        tp        = round(min(tp, swing_tp + atr14 * 0.2), 2)

    return sl, tp, abs(sl_dist)


def compute_position(sl_dist):
    risk_usd   = ACCOUNT_USD * RISK_PCT
    pos_oz     = risk_usd / sl_dist if sl_dist > 0 else 0
    reward_usd = pos_oz * sl_dist * RR_RATIO
    return round(pos_oz, 4), round(risk_usd, 2), round(reward_usd, 2)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print(f"  GOLD (GC=F) — 15-Min Trade Planner")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 62)

    df      = fetch_gold()
    cur_px  = float(df["close"].iloc[-1])
    atr14   = float(atr(df, 14).iloc[-1])
    ema20_v = float(ema(df["close"], 20).iloc[-1])
    ema50_v = float(ema(df["close"], 50).iloc[-1])
    rsi14   = float(rsi(df["close"], 14).iloc[-1])
    s_high  = swing_high(df, SWING_BARS)
    s_low   = swing_low(df, SWING_BARS)

    print(f"\n  Last bar  : {df.index[-1].strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Price     : ${cur_px:,.3f}")
    print(f"  ATR(14)   : ${atr14:.3f}  |  EMA20: ${ema20_v:,.3f}  |  EMA50: ${ema50_v:,.3f}")
    print(f"  RSI(14)   : {rsi14:.1f}   |  Swing High: ${s_high:,.3f}  |  Swing Low: ${s_low:,.3f}")

    # ── Forecast ───────────────────────────────────────────────
    print(f"\n  Running AutoTheta + AutoETS ({TRAIN_BARS} bars training) ...")
    max_h           = max(HORIZONS)
    theta, ets, ens = run_forecast(df, max_h)
    direction       = "SELL" if ens[0] < 0 else "BUY"
    arrow           = "v" if direction == "SELL" else "^"

    print(f"\n  {'Horizon':<10} {'AutoTheta':>12} {'AutoETS':>12} {'Ensemble':>12}  Dir")
    print("  " + "-" * 56)
    for h in HORIZONS:
        cum_ens = sum(ens[:h])
        pred_px = cur_px * np.exp(cum_ens)
        d       = "v" if ens[h-1] < 0 else "^"
        label   = f"+{h*15}min" if h < 4 else "+1hr"
        print(f"  {label:<10} ${cur_px*np.exp(sum(theta[:h])):>10,.3f}  "
              f"${cur_px*np.exp(sum(ets[:h])):>10,.3f}  ${pred_px:>10,.3f}   {d}")

    # ── Trade plan ─────────────────────────────────────────────
    entry, entry_type = compute_entry(df, direction, cur_px, atr14, ema20_v)
    sl, tp, sl_dist   = compute_sl_tp(df, direction, entry, atr14)
    pos_oz, risk_usd, reward_usd = compute_position(sl_dist)
    rr_actual = (abs(tp - entry) / sl_dist) if sl_dist > 0 else 0

    # RSI filter warnings
    rsi_warn = ""
    if direction == "SELL" and rsi14 < 30:
        rsi_warn = "  [!] RSI oversold — weaker SELL setup"
    elif direction == "BUY" and rsi14 > 70:
        rsi_warn = "  [!] RSI overbought — weaker BUY setup"

    print(f"\n{'=' * 62}")
    print(f"  TRADE PLAN  {direction} {arrow}  XAUUSD  [15min]")
    print(f"{'=' * 62}")
    print(f"  Entry type : {entry_type}")
    print(f"  Entry      : ${entry:>10,.3f}")
    print(f"  Stop Loss  : ${sl:>10,.3f}   ({sl_dist:.3f} pts = {ATR_SL_MULT}x ATR)")
    print(f"  Take Profit: ${tp:>10,.3f}   ({abs(tp-entry):.3f} pts)")
    print(f"  R:R        : 1:{rr_actual:.2f}")
    print(f"  " + "-"*53)
    print(f"  Account    : ${ACCOUNT_USD:,.0f}  |  Risk: {RISK_PCT:.1%} = ${risk_usd:.2f}")
    print(f"  Position   : {pos_oz:.4f} oz")
    print(f"  Max loss   : ${risk_usd:.2f}  |  Max gain: ${reward_usd:.2f}")
    if rsi_warn:
        print(rsi_warn)
    print(f"{'=' * 62}")

    # ── Context ─────────────────────────────────────────────────
    price_vs_ema = "BELOW" if cur_px < ema20_v else "ABOVE"
    ema_stack    = "EMA20 > EMA50 (bullish)" if ema20_v > ema50_v else "EMA20 < EMA50 (bearish)"
    print(f"\n  Context:")
    print(f"  Price {price_vs_ema} EMA20  |  {ema_stack}")
    print(f"  Swing range: ${s_low:,.3f} — ${s_high:,.3f}  (width: ${s_high-s_low:.2f})")
    print(f"  ATR regime : {'HIGH' if atr14 > 10 else 'NORMAL' if atr14 > 5 else 'LOW'} volatility")
    print()


if __name__ == "__main__":
    main()
