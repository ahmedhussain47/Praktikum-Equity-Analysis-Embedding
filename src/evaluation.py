from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Models whose predictions are always constant (e.g. always 0).
# Their cross-sectional ranking is arbitrary so IC/RankIC/Sharpe are excluded.
_CONSTANT_PRED_MODELS = {"Naive-RandomWalk"}

# Rebalancing frequency: 18 cutoffs over ~18 months ≈ 12 obs/year → annualise by √12
_SHARPE_ANNUAL_FACTOR = 12

# Long-short portfolio: top/bottom 20% of the ~160-asset universe
_LS_TOP_N    = 32
_LS_BOTTOM_N = 32


def directional_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def summarize_predictions(predictions):
    rows = []

    for (model, horizon), g in predictions.groupby(["Model", "Horizon"]):
        g = g.dropna(subset=["y_true", "y_pred"])

        if g.empty:
            continue

        mse = mean_squared_error(g["y_true"], g["y_pred"])
        rmse = mse ** 0.5

        rows.append({
            "Model": model,
            "Horizon": horizon,
            "N": len(g),
            "MAE": mean_absolute_error(g["y_true"], g["y_pred"]),
            "RMSE": rmse,
            "DirectionalAccuracy": directional_accuracy(g["y_true"], g["y_pred"]),
            "MeanActualReturn": g["y_true"].mean(),
            "MeanPredReturn": g["y_pred"].mean(),
        })

    return pd.DataFrame(rows).sort_values(["Horizon", "MAE", "Model"]).reset_index(drop=True)


def _normalize_cutoff(predictions: pd.DataFrame) -> pd.DataFrame:
    """Normalize cutoff_date to YYYY-MM-DD string to fix mixed-format duplicates."""
    predictions = predictions.copy()
    if "cutoff_date" in predictions.columns:
        predictions["cutoff_date"] = (
            pd.to_datetime(predictions["cutoff_date"], format="mixed")
            .dt.strftime("%Y-%m-%d")
        )
    return predictions


def cross_sectional_ic(predictions: pd.DataFrame, min_assets: int = 20) -> pd.DataFrame:
    """
    Compute per-cutoff Pearson IC and Spearman RankIC for each (model, horizon).
    Groups by cutoff_date (not Date) to get one cross-section per rebalancing period.
    Excludes models with constant predictions (e.g. Naive-RandomWalk).
    """
    predictions = _normalize_cutoff(predictions)
    group_col = "cutoff_date" if "cutoff_date" in predictions.columns else "Date"
    rows = []

    for (model, horizon), mg in predictions.groupby(["Model", "Horizon"]):
        if model in _CONSTANT_PRED_MODELS:
            continue
        for cutoff, g in mg.groupby(group_col):
            g = g.dropna(subset=["y_true", "y_pred"])
            if g["Ticker"].nunique() < min_assets:
                continue
            if g["y_pred"].nunique() < 2:   # constant predictions at this cutoff
                continue
            try:
                ic,     _ = pearsonr(g["y_pred"], g["y_true"])
                rank_ic, _ = spearmanr(g["y_pred"], g["y_true"])
            except Exception:
                continue
            if np.isfinite(ic) and np.isfinite(rank_ic):
                rows.append({
                    "Model": model, "Horizon": horizon, "Cutoff": cutoff,
                    "IC": float(ic), "RankIC": float(rank_ic),
                    "N_Assets": int(g["Ticker"].nunique()),
                })
    return pd.DataFrame(rows)


def summarize_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise per-cutoff IC/RankIC into mean IC, mean RankIC, and IR.
    IR = mean(IC) / std(IC)  — measures signal consistency across time.
    """
    if ic_df.empty:
        return pd.DataFrame(columns=["Model", "Horizon", "IC", "RankIC", "IR", "N_Cutoffs"])
    rows = []
    for (model, horizon), g in ic_df.groupby(["Model", "Horizon"]):
        mean_ic  = g["IC"].mean()
        std_ic   = g["IC"].std(ddof=1)
        mean_ric = g["RankIC"].mean()
        rows.append({
            "Model":     model,
            "Horizon":   horizon,
            "IC":        round(mean_ic,  4),
            "RankIC":    round(mean_ric, 4),
            "IR":        round(mean_ic / std_ic, 4) if std_ic > 0 else np.nan,
            "N_Cutoffs": len(g),
        })
    return pd.DataFrame(rows).sort_values(["Horizon", "IC"], ascending=[True, False]).reset_index(drop=True)


# Keep old name as alias so existing notebook calls still work
def cross_sectional_rank_ic(predictions: pd.DataFrame, min_assets: int = 20) -> pd.DataFrame:
    return cross_sectional_ic(predictions, min_assets)


def summarize_rank_ic(rank_ic_df: pd.DataFrame) -> pd.DataFrame:
    return summarize_ic(rank_ic_df)


def long_short_portfolio_summary(
    predictions: pd.DataFrame,
    top_n: int = _LS_TOP_N,
    bottom_n: int = _LS_BOTTOM_N,
) -> pd.DataFrame:
    """
    Cross-sectional long-short portfolio back-test.

    For each (model, horizon, cutoff_date):
      - Rank all assets by y_pred (descending).
      - Long top_n assets (20% of ~160), short bottom_n assets.
      - Portfolio return = mean(long y_true) − mean(short y_true).

    Sharpe is annualised using sqrt(12) because the portfolio rebalances monthly
    (one cutoff every 21 trading days ≈ 12 rebalances per year).

    Models with constant predictions (Naive-RandomWalk) are excluded:
    their all-zero predictions produce an arbitrary ranking that is not a
    real signal and yields a spurious non-zero Sharpe.
    """
    predictions = _normalize_cutoff(predictions)
    group_col = "cutoff_date" if "cutoff_date" in predictions.columns else "Date"

    rows = []
    for (model, horizon), mg in predictions.groupby(["Model", "Horizon"]):
        if model in _CONSTANT_PRED_MODELS:
            continue
        for cutoff, g in mg.groupby(group_col):
            g = g.dropna(subset=["y_true", "y_pred"]).copy()
            if len(g) < top_n + bottom_n:
                continue
            if g["y_pred"].nunique() < 2:   # constant preds → arbitrary rank
                continue
            ranked     = g.sort_values("y_pred", ascending=False)
            long_ret   = ranked.head(top_n)["y_true"].mean()
            short_ret  = ranked.tail(bottom_n)["y_true"].mean()
            rows.append({
                "Model": model, "Horizon": horizon, "Cutoff": cutoff,
                "LongReturn": long_ret, "ShortReturn": short_ret,
                "LongShortReturn": long_ret - short_ret,
                "N_Assets": g["Ticker"].nunique(),
            })

    per_cutoff = pd.DataFrame(rows)
    if per_cutoff.empty:
        return pd.DataFrame(columns=[
            "Model", "Horizon", "MeanLSReturn", "VolLSReturn", "Sharpe", "N_Cutoffs"
        ])

    summary_rows = []
    for (model, horizon), g in per_cutoff.groupby(["Model", "Horizon"]):
        pr       = g["LongShortReturn"].values
        mean_ret = pr.mean()
        vol_ret  = pr.std(ddof=1)
        sharpe   = (mean_ret / vol_ret * np.sqrt(_SHARPE_ANNUAL_FACTOR)
                    if vol_ret > 0 and np.isfinite(vol_ret) else np.nan)
        summary_rows.append({
            "Model":       model,
            "Horizon":     horizon,
            "MeanLSReturn": round(mean_ret, 6),
            "VolLSReturn":  round(vol_ret,  6),
            "Sharpe":       round(sharpe, 4) if np.isfinite(sharpe) else np.nan,
            "N_Cutoffs":    len(g),
        })

    return pd.DataFrame(summary_rows).sort_values(
        ["Horizon", "Sharpe"], ascending=[True, False]
    ).reset_index(drop=True)