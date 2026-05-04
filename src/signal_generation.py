"""
Forex Signal Generation Pipeline
Uses AutoTheta (h=5, RankIC=0.256) on 20 forex pairs.
Excludes commodities (=F) and gold (GC=F).
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "new_models"
SIGNALS_DIR = RESULTS_DIR / "signals"
SIGNALS_DIR.mkdir(exist_ok=True)

SIGNAL_MODEL = "AutoTheta"   # Best RankIC on forex h=5: 0.256
SIGNAL_HORIZON = 5           # Weekly rebalancing
TOP_N = 4                    # Long top 4 pairs (20%)
BOTTOM_N = 4                 # Short bottom 4 pairs (20%)
ANNUAL_FACTOR = 52           # Weekly → annual Sharpe


def load_forex_predictions(model: str = SIGNAL_MODEL, horizon: int = SIGNAL_HORIZON) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "all_predictions.csv")
    forex = df[
        df["Ticker"].str.endswith("=X") &   # forex pairs only
        (df["Model"] == model) &
        (df["Horizon"] == horizon)
    ].copy()
    forex["Date"] = pd.to_datetime(forex["Date"])
    return forex


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cutoff date, rank all forex pairs by predicted return.
    Signal: +1 (long) for top N, -1 (short) for bottom N, 0 otherwise.
    """
    signals = []
    for cutoff in sorted(df["cutoff_date"].unique()):
        cut_df = df[df["cutoff_date"] == cutoff].copy()
        cut_df = cut_df.sort_values("y_pred", ascending=False).reset_index(drop=True)
        cut_df["rank"] = cut_df.index + 1
        n = len(cut_df)

        def assign_signal(rank):
            if rank <= TOP_N:
                return 1       # Long
            elif rank > n - BOTTOM_N:
                return -1      # Short
            return 0

        cut_df["signal"] = cut_df["rank"].apply(assign_signal)
        cut_df["return"] = cut_df["signal"] * cut_df["y_true"]
        signals.append(cut_df)

    return pd.concat(signals, ignore_index=True)


def compute_performance(signals: pd.DataFrame) -> dict:
    """Portfolio performance metrics from signal returns."""
    portfolio = signals.groupby("cutoff_date")["return"].mean()  # equal-weight L/S
    total_return = (1 + portfolio).prod() - 1
    sharpe = portfolio.mean() / portfolio.std() * np.sqrt(ANNUAL_FACTOR) if portfolio.std() > 0 else 0
    win_rate = (portfolio > 0).mean()
    max_dd = (portfolio.cumsum() - portfolio.cumsum().cummax()).min()

    return {
        "Total Return": f"{total_return:.2%}",
        "Annualized Sharpe": f"{sharpe:.2f}",
        "Win Rate (per period)": f"{win_rate:.1%}",
        "Max Drawdown": f"{max_dd:.4f}",
        "Num Periods": len(portfolio),
    }


def compute_rankic(signals: pd.DataFrame) -> float:
    from scipy.stats import spearmanr
    rics = []
    for cutoff in signals["cutoff_date"].unique():
        c = signals[signals["cutoff_date"] == cutoff]
        if len(c) >= 5:
            r, _ = spearmanr(c["y_pred"], c["y_true"])
            if not np.isnan(r):
                rics.append(r)
    return float(np.mean(rics)) if rics else np.nan


def save_signal_table(signals: pd.DataFrame):
    """Save a clean signal table: Date, Ticker, Predicted Return, Signal, Actual Return."""
    table = signals[["Date", "cutoff_date", "Ticker", "y_pred", "signal", "y_true"]].copy()
    table.columns = ["Date", "Cutoff", "Pair", "Pred_Return", "Signal", "Actual_Return"]
    table["Direction"] = table["Signal"].map({1: "LONG", -1: "SHORT", 0: "NEUTRAL"})
    table = table[table["Signal"] != 0]  # only actionable signals
    table.to_csv(SIGNALS_DIR / "forex_signals.csv", index=False)
    print(f"Signals saved: {SIGNALS_DIR / 'forex_signals.csv'}")
    return table


def plot_cumulative_pnl(signals: pd.DataFrame):
    import matplotlib.pyplot as plt

    portfolio = signals.groupby("cutoff_date")["return"].mean()
    cumulative = (1 + portfolio).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(range(len(cumulative)), cumulative.values, color="steelblue", linewidth=2)
    axes[0].axhline(1, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title(f"Forex Signal Cumulative PnL — {SIGNAL_MODEL} h={SIGNAL_HORIZON}\n(Long top {TOP_N} / Short bottom {TOP_N} of 20 pairs, weekly rebalance)", fontsize=13)
    axes[0].set_ylabel("Portfolio Value (start=1)")
    axes[0].set_xlabel("Rebalancing Period")
    axes[0].grid(alpha=0.3)

    axes[1].bar(range(len(portfolio)), portfolio.values,
                color=["green" if r > 0 else "red" for r in portfolio.values], alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Per-Period Portfolio Return", fontsize=11)
    axes[1].set_ylabel("Return")
    axes[1].set_xlabel("Rebalancing Period")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = SIGNALS_DIR / "forex_pnl.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PnL chart saved: {out}")


def show_latest_signals(signals: pd.DataFrame):
    """Show most recent signal for each pair (latest cutoff)."""
    latest_cutoff = signals["cutoff_date"].max()
    latest = signals[signals["cutoff_date"] == latest_cutoff].sort_values("y_pred", ascending=False)
    print(f"\n=== LATEST SIGNALS (cutoff: {latest_cutoff}) ===")
    print(f"{'Pair':<15} {'Pred Return':>12} {'Signal':>8} {'Actual':>10}")
    print("-" * 50)
    for _, row in latest.iterrows():
        sig_label = "LONG  ^" if row["signal"] == 1 else ("SHORT v" if row["signal"] == -1 else "  ---  ")
        print(f"{row['Ticker']:<15} {row['y_pred']:>12.5f} {sig_label:>8} {row['y_true']:>10.5f}")


if __name__ == "__main__":
    print(f"=== FOREX SIGNAL GENERATION ===")
    print(f"Model: {SIGNAL_MODEL} | Horizon: h={SIGNAL_HORIZON} (weekly) | Pairs: 20 forex (no gold/commodities)\n")

    df = load_forex_predictions()
    print(f"Loaded {len(df)} forex predictions across {df['Ticker'].nunique()} pairs")

    signals = generate_signals(df)

    rankic = compute_rankic(signals)
    perf = compute_performance(signals)

    print("\n=== SIGNAL PERFORMANCE ===")
    print(f"RankIC: {rankic:.4f}")
    for k, v in perf.items():
        print(f"{k}: {v}")

    show_latest_signals(signals)
    table = save_signal_table(signals)
    plot_cumulative_pnl(signals)

    print("\nDone.")
