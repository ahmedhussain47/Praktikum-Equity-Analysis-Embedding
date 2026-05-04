"""
data_feeds.py — Multi-timeframe OHLCV ingestion for Gold, Forex, and Equities.

Primary backend: yfinance (free, no API key).
4H bars are built by resampling 1H data (yfinance has no native 4H interval).
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Symbol mapping ─────────────────────────────────────────────────────────────
# Maps standard trading symbols → yfinance ticker strings.
SYMBOL_MAP: Dict[str, str] = {
    # Metals
    "XAUUSD": "GC=F",       # Gold futures (most liquid)
    "XAGUSD": "SI=F",       # Silver futures
    # Forex majors
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    # Forex crosses
    "GBPJPY": "GBPJPY=X",
    "EURJPY": "EURJPY=X",
    "EURGBP": "EURGBP=X",
}

# yfinance interval strings
_YF_INTERVAL: Dict[str, str] = {
    "1min":  "1m",    # yfinance max lookback: 7 days
    "5min":  "5m",
    "15min": "15m",
    "30min": "30m",
    "1H":    "1h",
    "4H":    "1h",   # fetched as 1H, then resampled
    "1D":    "1d",
}

# How far back to fetch for each timeframe
_FETCH_PERIOD: Dict[str, str] = {
    "1min":  "7d",    # yfinance hard limit for 1m bars
    "5min":  "5d",
    "15min": "30d",
    "30min": "30d",
    "1H":    "60d",
    "4H":    "90d",
    "1D":    "730d",
}

# Minimum bars required before we consider data usable
_MIN_BARS: Dict[str, int] = {
    "1min":  100,
    "5min":  50,
    "15min": 50,
    "30min": 50,
    "1H":    50,
    "4H":    30,
    "1D":    30,
}


def _to_yf_symbol(symbol: str) -> str:
    """Return the yfinance ticker string for a given symbol."""
    return SYMBOL_MAP.get(symbol.upper(), symbol)


def _resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1H OHLCV bars into 4H bars."""
    resampled = df.resample("4h", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open", "close"])
    return resampled


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    bars: int = 200,
    retries: int = 2,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for *symbol* at *timeframe* resolution.

    Args:
        symbol:    Standard symbol (e.g. "XAUUSD", "EURUSD", "AAPL").
        timeframe: One of "5min" | "15min" | "30min" | "1H" | "4H" | "1D".
        bars:      Maximum number of bars to return (most-recent).
        retries:   Number of retry attempts on network failure.

    Returns:
        DataFrame indexed by UTC timestamp with columns
        [open, high, low, close, volume]. Empty on failure.
    """
    yf_symbol   = _to_yf_symbol(symbol)
    yf_interval = _YF_INTERVAL.get(timeframe, "1h")
    period      = _FETCH_PERIOD.get(timeframe, "60d")

    for attempt in range(retries + 1):
        try:
            ticker = yf.Ticker(yf_symbol)
            raw = ticker.history(
                period=period,
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,
            )

            if raw.empty:
                logger.warning("No data: %s (%s) @ %s", symbol, yf_symbol, timeframe)
                return pd.DataFrame()

            # Normalise column names
            raw = raw.rename(columns={
                "Open": "open", "High": "high",
                "Low": "low",  "Close": "close", "Volume": "volume",
            })
            raw = raw[["open", "high", "low", "close", "volume"]].copy()

            # Ensure UTC-aware DatetimeIndex
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC")
            else:
                raw.index = raw.index.tz_convert("UTC")
            raw.index.name = "timestamp"

            if timeframe == "4H":
                raw = _resample_to_4h(raw)

            df = raw.tail(bars)

            min_bars = _MIN_BARS.get(timeframe, 30)
            if len(df) < min_bars:
                logger.warning(
                    "Insufficient bars for %s @ %s: got %d, need %d",
                    symbol, timeframe, len(df), min_bars,
                )
                return pd.DataFrame()

            logger.debug("Fetched %d bars for %s @ %s", len(df), symbol, timeframe)
            return df

        except Exception as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                logger.error("fetch_ohlcv failed for %s @ %s: %s", symbol, timeframe, exc)
                return pd.DataFrame()

    return pd.DataFrame()


def fetch_multi_timeframe(
    symbol: str,
    timeframes: List[str],
    bars: int = 300,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for several timeframes in sequence.

    Returns:
        Dict mapping timeframe → OHLCV DataFrame.
        Timeframes that fail are omitted (not None).
    """
    result: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        df = fetch_ohlcv(symbol, tf, bars=bars)
        if not df.empty:
            result[tf] = df
        else:
            logger.warning("Skipping %s @ %s — no usable data", symbol, tf)
    return result


# ── Live data feed with TTL cache ──────────────────────────────────────────────

_REFRESH_TTL: Dict[str, int] = {   # seconds before a refresh is triggered
    "5min":  45,
    "15min": 90,
    "30min": 150,
    "1H":    270,
    "4H":    600,
    "1D":    3600,
}


class LiveDataFeed:
    """
    Thin caching wrapper around fetch_ohlcv.

    Stores the last fetched DataFrame per timeframe and only calls
    the API again once the TTL for that timeframe has elapsed.
    """

    def __init__(self, symbol: str, timeframes: List[str], bars: int = 300):
        self.symbol     = symbol
        self.timeframes = timeframes
        self.bars       = bars
        self._cache: Dict[str, pd.DataFrame] = {}
        self._last:  Dict[str, float] = {}

    def _is_stale(self, tf: str) -> bool:
        last = self._last.get(tf, 0.0)
        ttl  = _REFRESH_TTL.get(tf, 300)
        return (time.time() - last) > ttl

    def get(self, timeframe: str) -> pd.DataFrame:
        """Return cached data, refreshing if stale."""
        if self._is_stale(timeframe):
            df = fetch_ohlcv(self.symbol, timeframe, bars=self.bars)
            if not df.empty:
                self._cache[timeframe] = df
                self._last[timeframe]  = time.time()
        return self._cache.get(timeframe, pd.DataFrame())

    def get_all(self) -> Dict[str, pd.DataFrame]:
        """Return a fresh snapshot for every configured timeframe."""
        return {tf: self.get(tf) for tf in self.timeframes}

    def invalidate(self, timeframe: Optional[str] = None):
        """Force a refresh on the next get() call."""
        if timeframe:
            self._last.pop(timeframe, None)
        else:
            self._last.clear()
