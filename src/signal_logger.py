"""
signal_logger.py — Persistent signal log and outcome tracker.

Signals are appended to a CSV file so they can be reviewed, audited,
and used to measure live accuracy over time.

Usage
─────
    logger = SignalLogger()
    logger.log(signal)                    # Record a new signal
    logger.record_outcome("2025-01-01 10:00", "TP", exit_price=2370.0)
    df = logger.load()                    # Load entire log as DataFrame
    print(logger.live_accuracy())         # Win-rate on resolved signals
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .signal_engine import Signal


# ── Column schema ──────────────────────────────────────────────────────────────

_SIGNAL_COLS = [
    "timestamp", "asset", "timeframe", "signal",
    "entry", "take_profit", "stop_loss",
    "confidence", "rr_ratio", "atr_value",
    "valid_until", "model_pred",
    "outcome",      # Filled in later: "TP" | "SL" | "EXPIRED" | ""
    "exit_price",   # Actual exit price
    "pnl_pct",      # Realised PnL %
]


class SignalLogger:
    """
    Append-only CSV log for generated signals and their outcomes.

    Parameters
    ──────────
    log_path:  Path to the CSV log file.
               Defaults to results/signals_log.csv relative to project root.
    """

    def __init__(self, log_path: Optional[str] = None):
        if log_path is None:
            here     = Path(__file__).resolve().parent
            log_path = here.parent / "results" / "signals_log.csv"

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_path.exists():
            self._write_header()

    # ── Write ───────────────────────────────────────────────────────────────────

    def _write_header(self):
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_SIGNAL_COLS)
            writer.writeheader()

    def log(self, signal: Signal) -> None:
        """Append a new signal to the log (outcome fields left blank)."""
        row = {
            "timestamp":   signal.timestamp,
            "asset":       signal.asset,
            "timeframe":   signal.timeframe,
            "signal":      signal.signal,
            "entry":       signal.entry,
            "take_profit": signal.take_profit,
            "stop_loss":   signal.stop_loss,
            "confidence":  signal.confidence,
            "rr_ratio":    signal.rr_ratio,
            "atr_value":   signal.atr_value,
            "valid_until": signal.valid_until,
            "model_pred":  signal.model_pred if signal.model_pred is not None else "",
            "outcome":     "",
            "exit_price":  "",
            "pnl_pct":     "",
        }
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_SIGNAL_COLS)
            writer.writerow(row)

        print(
            f"[SignalLogger] Logged {signal.signal} {signal.asset} @ {signal.entry:.5f} "
            f"(conf={signal.confidence}%  TP={signal.take_profit:.5f}  SL={signal.stop_loss:.5f})"
        )

    def record_outcome(
        self,
        timestamp:   str,
        outcome:     str,           # "TP" | "SL" | "EXPIRED"
        exit_price:  float,
        entry_price: Optional[float] = None,
        direction:   Optional[str]   = None,
    ) -> bool:
        """
        Update the outcome for a previously logged signal.

        Matches on timestamp string.  Re-writes the entire file in place
        (acceptable for a small CSV log).

        Returns True if the matching row was found and updated.
        """
        df = self.load()
        if df.empty:
            return False

        mask = df["timestamp"] == timestamp
        if not mask.any():
            print(f"[SignalLogger] No signal found for timestamp: {timestamp}")
            return False

        df.loc[mask, "outcome"]    = outcome
        df.loc[mask, "exit_price"] = exit_price

        # Compute pnl_pct if we have entry info
        if entry_price is not None and direction is not None and entry_price > 0:
            if direction == "BUY":
                pnl = (exit_price - entry_price) / entry_price * 100.0
            else:
                pnl = (entry_price - exit_price) / entry_price * 100.0
            df.loc[mask, "pnl_pct"] = round(pnl, 5)
        elif entry_price is None:
            # Try to use the logged entry price
            ep = df.loc[mask, "entry"].values[0]
            di = df.loc[mask, "signal"].values[0]
            if ep and di:
                pnl = (
                    (exit_price - float(ep)) / float(ep) * 100.0
                    if di == "BUY"
                    else (float(ep) - exit_price) / float(ep) * 100.0
                )
                df.loc[mask, "pnl_pct"] = round(pnl, 5)

        df.to_csv(self.log_path, index=False)
        print(f"[SignalLogger] Recorded outcome: {outcome} @ {exit_price}")
        return True

    # ── Read ────────────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Return the full signal log as a DataFrame."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return pd.DataFrame(columns=_SIGNAL_COLS)

        df = pd.read_csv(self.log_path, dtype=str)
        # Parse numeric columns
        for col in ["entry", "take_profit", "stop_loss", "confidence",
                    "rr_ratio", "atr_value", "exit_price", "pnl_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ── Analytics ───────────────────────────────────────────────────────────────

    def live_accuracy(self) -> dict:
        """
        Compute win-rate and expectancy from signals that have been resolved.
        """
        df = self.load()
        resolved = df[df["outcome"].isin(["TP", "SL", "EXPIRED"])].copy()

        if resolved.empty:
            return {"resolved_trades": 0, "message": "No resolved signals yet"}

        n_total   = len(resolved)
        n_tp      = (resolved["outcome"] == "TP").sum()
        n_sl      = (resolved["outcome"] == "SL").sum()
        n_expired = (resolved["outcome"] == "EXPIRED").sum()
        win_rate  = n_tp / n_total

        pnl = resolved["pnl_pct"].dropna()
        avg_pnl    = pnl.mean() if not pnl.empty else float("nan")
        total_pnl  = pnl.sum()  if not pnl.empty else float("nan")

        # Stratify by asset
        by_asset = {}
        for asset, grp in resolved.groupby("asset"):
            wr = (grp["outcome"] == "TP").mean()
            by_asset[asset] = {
                "trades":   len(grp),
                "win_rate": f"{wr:.1%}",
            }

        return {
            "resolved_trades": n_total,
            "tp_count":        int(n_tp),
            "sl_count":        int(n_sl),
            "expired_count":   int(n_expired),
            "win_rate":        f"{win_rate:.1%}",
            "avg_pnl_pct":     f"{avg_pnl:.3f}%",
            "total_pnl_pct":   f"{total_pnl:.2f}%",
            "by_asset":        by_asset,
        }

    def export_json(self, path: Optional[str] = None) -> str:
        """Export the log to a JSON file and return the path."""
        df    = self.load()
        out   = path or str(self.log_path.with_suffix(".json"))
        df.to_json(out, orient="records", indent=2)
        print(f"[SignalLogger] Exported {len(df)} records to {out}")
        return out

    def __len__(self) -> int:
        return len(self.load())

    def __repr__(self) -> str:
        return f"SignalLogger(path='{self.log_path}', records={len(self)})"
