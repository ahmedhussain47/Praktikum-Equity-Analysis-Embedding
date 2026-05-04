"""
backtester.py — Walk-forward signal backtesting.

Simulation rules
────────────────
• Entry is on the NEXT bar's OPEN after a signal is generated
  (no look-ahead bias — we cannot trade on the signal bar's close).
• On each subsequent bar we check:
    BUY : SL hit if low ≤ stop_loss   |  TP hit if high ≥ take_profit
    SELL: SL hit if high ≥ stop_loss  |  TP hit if low  ≤ take_profit
• If BOTH TP and SL are touched on the same bar, SL is assumed hit first
  (conservative approach).
• Slippage and commission are applied symmetrically to every trade.
• Expired trades (no TP/SL within max_bars) exit at the last available close.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .signal_engine import Signal


# ── Trade result dataclass ─────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    signal_time:  str
    entry_time:   Optional[pd.Timestamp]
    exit_time:    Optional[pd.Timestamp]
    asset:        str
    direction:    str
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    exit_price:   float
    outcome:      str        # "TP" | "SL" | "EXPIRED"
    bars_held:    int
    pnl_pct:      float      # Net return (%) after slippage + commission
    rr_achieved:  float      # Actual R achieved (negative when loss)
    confidence:   int
    rr_planned:   float


# ── Backtester ─────────────────────────────────────────────────────────────────

class Backtester:
    """
    Simulates a list of Signal objects on historical OHLCV data.

    Parameters
    ──────────
    slippage_pct:    One-way slippage as fraction of price (0.0001 = 1 pip on 1.0 FX).
    commission_pct:  Round-trip commission as fraction of price.
    max_bars:        Maximum bars before a trade is closed at market (expiry).
    """

    def __init__(
        self,
        slippage_pct:   float = 0.0001,
        commission_pct: float = 0.0001,
        max_bars:       int   = 100,
    ):
        self.slippage_pct   = slippage_pct
        self.commission_pct = commission_pct
        self.max_bars       = max_bars
        self._trades: List[BacktestTrade] = []

    # ── Public API ──────────────────────────────────────────────────────────────

    def run(
        self,
        signals:    List[Signal],
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Simulate all signals and return a DataFrame of trade results.

        Args:
            signals:    List of Signal objects from SignalEngine.scan_history().
            price_data: OHLCV DataFrame at the entry timeframe resolution.
                        Must cover at least max_bars beyond the last signal.

        Returns:
            DataFrame with one row per trade.
        """
        self._trades = []

        for sig in signals:
            trade = self._simulate_one(sig, price_data)
            if trade is not None:
                self._trades.append(trade)

        return self.results_df()

    def results_df(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self._trades])

    # ── Internal simulation ────────────────────────────────────────────────────

    def _simulate_one(
        self,
        sig:   Signal,
        ohlcv: pd.DataFrame,
    ) -> Optional[BacktestTrade]:
        """Simulate a single trade from entry to exit."""
        sig_ts = pd.Timestamp(sig.timestamp).tz_localize("UTC") \
                 if pd.Timestamp(sig.timestamp).tz is None \
                 else pd.Timestamp(sig.timestamp)

        # Find first bar AFTER the signal timestamp
        future = ohlcv[ohlcv.index > sig_ts]
        if future.empty:
            return None

        entry_bar    = future.iloc[0]
        entry_time   = future.index[0]
        slip         = sig.entry * self.slippage_pct

        # Slippage worsens entry
        if sig.signal == "BUY":
            actual_entry = sig.entry + slip
        else:
            actual_entry = sig.entry - slip

        # Recalculate TP/SL distances around the actual (post-slippage) entry
        sl_dist = abs(sig.entry - sig.stop_loss)
        tp_dist = abs(sig.take_profit - sig.entry)

        if sig.signal == "BUY":
            actual_sl = actual_entry - sl_dist
            actual_tp = actual_entry + tp_dist
        else:
            actual_sl = actual_entry + sl_dist
            actual_tp = actual_entry - tp_dist

        # Scan bar by bar for TP / SL touch
        subsequent = future.iloc[1:].head(self.max_bars)
        outcome, exit_price, exit_time, bars_held = self._scan_bars(
            sig.signal, actual_tp, actual_sl, subsequent
        )

        # Commission (round-trip, applied to entry amount)
        commission_cost = actual_entry * self.commission_pct * 2.0

        # PnL
        if sig.signal == "BUY":
            raw_pnl = (exit_price - actual_entry) / actual_entry
        else:
            raw_pnl = (actual_entry - exit_price) / actual_entry

        pnl_pct = raw_pnl - commission_cost / actual_entry

        # R achieved
        if sl_dist > 0:
            r = abs(exit_price - actual_entry) / sl_dist
            rr_achieved = r if pnl_pct > 0 else -r
        else:
            rr_achieved = 0.0

        return BacktestTrade(
            signal_time  = sig.timestamp,
            entry_time   = entry_time,
            exit_time    = exit_time,
            asset        = sig.asset,
            direction    = sig.signal,
            entry_price  = round(actual_entry, 5),
            stop_loss    = round(actual_sl, 5),
            take_profit  = round(actual_tp, 5),
            exit_price   = round(exit_price, 5),
            outcome      = outcome,
            bars_held    = bars_held,
            pnl_pct      = round(pnl_pct * 100.0, 5),  # as %
            rr_achieved  = round(rr_achieved, 3),
            confidence   = sig.confidence,
            rr_planned   = sig.rr_ratio,
        )

    def _scan_bars(
        self,
        direction: str,
        tp:        float,
        sl:        float,
        bars:      pd.DataFrame,
    ):
        """Iterate bars to find first exit event. Returns (outcome, price, time, count)."""
        for n, (ts, row) in enumerate(bars.iterrows(), start=1):
            if direction == "BUY":
                sl_hit = row["low"]  <= sl
                tp_hit = row["high"] >= tp
            else:
                sl_hit = row["high"] >= sl
                tp_hit = row["low"]  <= tp

            # Conservative: SL wins if both hit on the same bar
            if sl_hit:
                return "SL", sl, ts, n
            if tp_hit:
                return "TP", tp, ts, n

        # No exit within max_bars — close at last available close
        if not bars.empty:
            last_ts    = bars.index[-1]
            last_close = float(bars["close"].iloc[-1])
            return "EXPIRED", last_close, last_ts, len(bars)

        return "EXPIRED", sl, None, 0

    # ── Performance analytics ──────────────────────────────────────────────────

    def performance_summary(self) -> Dict:
        """
        Compute a comprehensive performance report from simulated trades.

        Returns a dict with:
            win_rate, avg_win, avg_loss, profit_factor, expectancy,
            sharpe_ratio, max_drawdown, total_return, by_confidence
        """
        if not self._trades:
            return {"error": "No trades to analyse"}

        df = self.results_df()

        wins    = df[df["outcome"] == "TP"]
        losses  = df[df["outcome"] == "SL"]
        expired = df[df["outcome"] == "EXPIRED"]

        pnl     = df["pnl_pct"].values
        n_total = len(df)

        win_rate = len(wins) / n_total if n_total else 0.0
        avg_win  = wins["pnl_pct"].mean()  if len(wins)   else 0.0
        avg_loss = losses["pnl_pct"].mean() if len(losses) else 0.0

        gross_profit = wins["pnl_pct"].sum()
        gross_loss   = abs(losses["pnl_pct"].sum())
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        expectancy    = float(np.mean(pnl))
        total_return  = float(np.sum(pnl))

        # Annualised Sharpe (assume ~252 trading periods per year at entry TF)
        pnl_std = float(np.std(pnl, ddof=1))
        sharpe  = (
            (expectancy / pnl_std * np.sqrt(252)) if pnl_std > 0 else 0.0
        )

        # Max drawdown on cumulative PnL curve
        cum = np.cumsum(pnl)
        peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))
        dd   = peak[1:] - cum
        max_dd = float(dd.max()) if len(dd) else 0.0

        return {
            "total_trades":       n_total,
            "tp_count":           len(wins),
            "sl_count":           len(losses),
            "expired_count":      len(expired),
            "win_rate":           f"{win_rate:.1%}",
            "avg_win_pct":        f"{avg_win:.3f}%",
            "avg_loss_pct":       f"{avg_loss:.3f}%",
            "profit_factor":      round(profit_factor, 2),
            "expectancy_pct":     f"{expectancy:.3f}%",
            "total_return_pct":   f"{total_return:.2f}%",
            "sharpe_ratio":       round(sharpe, 2),
            "max_drawdown_pct":   f"{max_dd:.2f}%",
            "avg_bars_held":      round(df["bars_held"].mean(), 1),
            "by_confidence":      self._by_confidence(df),
            "by_direction":       self._by_direction(df),
        }

    def _by_confidence(self, df: pd.DataFrame) -> Dict:
        """Win-rate stratified by confidence bucket."""
        buckets = [(60, 70), (70, 80), (80, 90), (90, 101)]
        result  = {}
        for lo, hi in buckets:
            sub = df[(df["confidence"] >= lo) & (df["confidence"] < hi)]
            if len(sub) == 0:
                continue
            wr  = (sub["outcome"] == "TP").mean()
            key = f"{lo}–{min(hi-1, 100)}%"
            result[key] = {
                "trades":    int(len(sub)),
                "win_rate":  f"{wr:.1%}",
                "avg_pnl":   f"{sub['pnl_pct'].mean():.3f}%",
            }
        return result

    def _by_direction(self, df: pd.DataFrame) -> Dict:
        """Win-rate split by BUY vs SELL."""
        result = {}
        for d in ["BUY", "SELL"]:
            sub = df[df["direction"] == d]
            if len(sub) == 0:
                continue
            wr = (sub["outcome"] == "TP").mean()
            result[d] = {
                "trades":   int(len(sub)),
                "win_rate": f"{wr:.1%}",
                "avg_pnl":  f"{sub['pnl_pct'].mean():.3f}%",
            }
        return result

    # ── Equity curve helper ────────────────────────────────────────────────────

    def equity_curve(self, starting_equity: float = 10_000.0) -> pd.Series:
        """
        Compound PnL into a USD equity curve.

        Each trade's pnl_pct is compounded onto the running balance.
        Returns a pd.Series indexed by trade number.
        """
        df = self.results_df()
        if df.empty:
            return pd.Series(dtype=float)

        equity = [starting_equity]
        for pnl in df["pnl_pct"]:
            equity.append(equity[-1] * (1.0 + pnl / 100.0))

        return pd.Series(equity, name="equity_usd")
