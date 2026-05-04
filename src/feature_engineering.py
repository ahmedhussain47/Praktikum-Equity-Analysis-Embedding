"""
feature_engineering.py — Technical indicator library for the signal engine.

All indicators are implemented with numpy/pandas only — no extra dependencies.
Each function is pure (no side effects) and operates on a pd.Series or
pd.DataFrame, returning a pd.Series unless noted.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Trend indicators ───────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period, min_periods=period).mean()


def dema(series: pd.Series, period: int) -> pd.Series:
    """Double EMA — faster-responding trend filter."""
    e = ema(series, period)
    return 2 * e - ema(e, period)


# ── Momentum indicators ────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta    = series.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD oscillator.

    Returns:
        (macd_line, signal_line, histogram)
    """
    fast_ema    = ema(series, fast)
    slow_ema    = ema(series, slow)
    macd_line   = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic oscillator (%K, %D).

    Args:
        df: DataFrame with columns 'high', 'low', 'close'.
    """
    lo  = df["low"].rolling(k_period, min_periods=k_period).min()
    hi  = df["high"].rolling(k_period, min_periods=k_period).max()
    k   = 100.0 * (df["close"] - lo) / (hi - lo).replace(0.0, np.nan)
    d   = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


# ── Volatility indicators ──────────────────────────────────────────────────────

def true_range(df: pd.DataFrame) -> pd.Series:
    """Raw True Range (not averaged)."""
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift(1)).abs()
    lc  = (df["low"]  - df["close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Returns:
        (upper_band, middle_band, lower_band)
    """
    mid   = sma(series, period)
    std   = series.rolling(period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ── Trend-strength indicators ──────────────────────────────────────────────────

def adx(
    df: pd.DataFrame,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index.

    Returns:
        (adx_line, plus_di, minus_di)
        adx_line: 0–100, >25 means trending market
        plus_di > minus_di: bullish pressure dominant
    """
    up_move   =  df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm  = np.where((up_move > down_move)   & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0)

    atr_vals = atr(df, period)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()

    plus_di  = 100.0 * plus_dm_s  / atr_vals.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm_s / atr_vals.replace(0.0, np.nan)

    di_sum  = (plus_di + minus_di).replace(0.0, np.nan)
    dx      = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_val = dx.ewm(alpha=1.0 / period, adjust=False).mean()

    return adx_val, plus_di, minus_di


# ── Support / Resistance ───────────────────────────────────────────────────────

def pivot_levels(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """
    Dynamic support/resistance using rolling high/low pivots.

    Returns the nearest resistance above and support below the last close.
    """
    close = float(df["close"].iloc[-1])
    window = df.tail(lookback * 2)

    # Swing highs: bar whose high is the highest in ±lookback bars
    highs = (
        window["high"]
        .rolling(lookback, center=True, min_periods=lookback // 2)
        .max()
    )
    lows = (
        window["low"]
        .rolling(lookback, center=True, min_periods=lookback // 2)
        .min()
    )

    pivot_highs = sorted(
        {round(v, 5) for v in highs.dropna().unique() if v > close}
    )
    pivot_lows = sorted(
        {round(v, 5) for v in lows.dropna().unique() if v < close},
        reverse=True,
    )

    atr_val = float(atr(df, 14).iloc[-1])
    return {
        "resistance": pivot_highs[0] if pivot_highs else close + atr_val * 2.0,
        "support":    pivot_lows[0]  if pivot_lows  else close - atr_val * 2.0,
    }


# ── Full feature matrix ────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append all technical indicators to an OHLCV DataFrame.

    Input  columns: open, high, low, close, volume
    Output columns: all of the above + indicator columns

    Rows with NaN (warm-up period) are dropped.
    """
    out = df.copy()
    c   = df["close"]

    # Trend
    out["ema_20"]  = ema(c, 20)
    out["ema_50"]  = ema(c, 50)
    out["ema_200"] = ema(c, 200)
    out["dema_20"] = dema(c, 20)

    # Momentum
    out["rsi_14"]                               = rsi(c, 14)
    out["rsi_7"]                                = rsi(c, 7)
    macd_l, macd_s, macd_h                      = macd(c)
    out["macd_line"]                            = macd_l
    out["macd_signal"]                          = macd_s
    out["macd_hist"]                            = macd_h
    out["stoch_k"], out["stoch_d"]              = stochastic(df)

    # Volatility
    out["atr_14"]                               = atr(df, 14)
    out["atr_7"]                                = atr(df,  7)
    bb_upper, bb_mid, bb_lower                  = bollinger_bands(c, 20, 2.0)
    out["bb_upper"]                             = bb_upper
    out["bb_mid"]                               = bb_mid
    out["bb_lower"]                             = bb_lower
    out["bb_width"]                             = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    # %B: 0 = at lower band, 0.5 = mid, 1 = at upper band
    out["bb_pct"]                               = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # Trend strength
    adx_v, plus_di_v, minus_di_v               = adx(df, 14)
    out["adx_14"]                               = adx_v
    out["plus_di"]                              = plus_di_v
    out["minus_di"]                             = minus_di_v

    # Price relative to moving averages (%)
    out["dist_ema20"]  = (c / out["ema_20"].replace(0, np.nan)  - 1) * 100
    out["dist_ema50"]  = (c / out["ema_50"].replace(0, np.nan)  - 1) * 100
    out["dist_ema200"] = (c / out["ema_200"].replace(0, np.nan) - 1) * 100

    # Returns
    out["ret_1"]   = c.pct_change(1)
    out["ret_5"]   = c.pct_change(5)
    out["ret_20"]  = c.pct_change(20)

    # Volatility regime (current ATR vs 50-bar average)
    out["vol_regime"] = out["atr_14"] / out["atr_14"].rolling(50, min_periods=10).mean()

    # Volume spike (only meaningful for assets with real volume)
    if "volume" in df.columns and df["volume"].sum() > 0:
        vol_ma = df["volume"].rolling(20, min_periods=5).mean()
        out["volume_ratio"] = df["volume"] / vol_ma.replace(0, np.nan)
    else:
        out["volume_ratio"] = 1.0

    return out.dropna(subset=["ema_200", "rsi_14", "atr_14"])


def build_multi_tf_snapshot(
    tf_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.Series]:
    """
    Compute features for each timeframe and return the most-recent bar
    as a dict of Series, keyed by timeframe string.

    Timeframes with insufficient data or computation errors are omitted.
    """
    snapshots: Dict[str, pd.Series] = {}
    for tf, df in tf_data.items():
        if df.empty or len(df) < 30:
            logger_msg = "Skipping %s: only %d bars"
            continue
        try:
            featured = compute_features(df)
            if not featured.empty:
                snapshots[tf] = featured.iloc[-1]
        except Exception:
            pass
    return snapshots


# ── Feature vector for ML model ────────────────────────────────────────────────

SIGNAL_FEATURE_COLS: List[str] = [
    "rsi_14", "rsi_7",
    "macd_hist", "macd_line", "macd_signal",
    "stoch_k", "stoch_d",
    "bb_pct", "bb_width",
    "adx_14", "plus_di", "minus_di",
    "dist_ema20", "dist_ema50", "dist_ema200",
    "ret_1", "ret_5", "vol_regime", "volume_ratio",
]


def extract_signal_features(featured_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extract the model-input feature matrix from a featured DataFrame.
    Returns only rows where all signal features are non-NaN.
    """
    cols = [c for c in SIGNAL_FEATURE_COLS if c in featured_df.columns]
    if not cols:
        return None
    return featured_df[cols].dropna()
