from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_utils import make_cutoff_dates, to_neuralforecast_df


@dataclass
class BenchmarkConfig:
    horizons: list[int]
    input_size: int
    test_step: int
    max_test_dates: int
    max_steps: int
    freq: str = "B"
    random_seed: int = 42


def _actual_close_at(panel: pd.DataFrame, ticker: str, cutoff_date: pd.Timestamp, horizon: int) -> tuple[pd.Timestamp, float, float] | None:
    g = panel[panel["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    matches = g.index[g["Date"] == cutoff_date].tolist()
    if not matches:
        return None
    idx = matches[0]
    target_idx = idx + horizon
    if target_idx >= len(g):
        return None
    last_close = float(g.loc[idx, "close"])
    actual_close = float(g.loc[target_idx, "close"])
    target_date = pd.Timestamp(g.loc[target_idx, "Date"])
    return target_date, last_close, actual_close


def run_neuralforecast_models(panel: pd.DataFrame, config: BenchmarkConfig, models_to_run: Iterable[str] = ("NBEATS", "PatchTST")) -> pd.DataFrame:
    """Train N-BEATS and/or PatchTST in a shared walk-forward setup.

    The model forecasts close prices. We convert close forecasts to future returns:
    predicted_return = forecast_close_h / cutoff_close - 1.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS, PatchTST

    all_predictions = []
    nf_all = to_neuralforecast_df(panel)
    tickers = sorted(panel["Ticker"].unique())

    for horizon in config.horizons:
        cutoff_dates = make_cutoff_dates(panel, config.input_size, horizon, config.test_step, config.max_test_dates)
        print(f"Horizon {horizon}: {len(cutoff_dates)} cutoff dates")

        for cutoff_date in cutoff_dates:
            train_df = nf_all[nf_all["ds"] <= cutoff_date].copy()
            train_df = train_df.sort_values(["unique_id", "ds"])

            model_objects = []
            if "NBEATS" in models_to_run:
                model_objects.append(
                    NBEATS(
                        h=horizon,
                        input_size=config.input_size,
                        # Important: NeuralForecast default N-BEATS uses trend/seasonality stacks,
                        # which fail for h=1. Identity stack works for h=1 and h=5.
                        stack_types=["identity"],
                        n_blocks=[2],
                        mlp_units=[[128, 128]],
                        max_steps=config.max_steps,
                        val_check_steps=config.max_steps,
                        random_seed=config.random_seed,
                        scaler_type="standard",
                        enable_progress_bar=False,
                        logger=False,
                    )
                )
            if "PatchTST" in models_to_run:
                model_objects.append(
                    PatchTST(
                        h=horizon,
                        input_size=config.input_size,
                        max_steps=config.max_steps,
                        val_check_steps=config.max_steps,
                        random_seed=config.random_seed,
                        patch_len=min(16, config.input_size // 2),
                        stride=min(8, config.input_size // 4),
                        enable_progress_bar=False,
                        logger=False,
                    )
                )

            if not model_objects:
                raise ValueError("No NeuralForecast models selected.")

            print(f"  Training {', '.join(models_to_run)} until {cutoff_date.date()}...")
            nf = NeuralForecast(models=model_objects, freq=config.freq)
            nf.fit(df=train_df, verbose=False)
            fcst = nf.predict(verbose=False).reset_index()

            # Keep the horizon-th row from each model forecast for each ticker.
            # NeuralForecast gives one row per future timestamp per unique_id.
            fcst = fcst.sort_values(["unique_id", "ds"])
            fcst["step"] = fcst.groupby("unique_id").cumcount() + 1
            fcst_h = fcst[fcst["step"] == horizon].copy()

            for _, row in fcst_h.iterrows():
                ticker = row["unique_id"]
                actual = _actual_close_at(panel, ticker, cutoff_date, horizon)
                if actual is None:
                    continue
                target_date, last_close, actual_close = actual

                for model_name in models_to_run:
                    if model_name not in fcst_h.columns:
                        continue
                    pred_close = float(row[model_name])
                    all_predictions.append({
                        "Date": target_date,
                        "Ticker": ticker,
                        "Model": model_name,
                        "Horizon": horizon,
                        "y_true": actual_close / last_close - 1.0,
                        "y_pred": pred_close / last_close - 1.0,
                        "cutoff_date": cutoff_date,
                        "last_close": last_close,
                        "pred_close": pred_close,
                        "actual_close": actual_close,
                    })

    return pd.DataFrame(all_predictions)


def run_naive_baseline(panel: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    """Create a Naive / Random Walk baseline in the same walk-forward format.

    The target used by the project is future return. The naive return forecast is:
        y_pred = 0

    This is equivalent to forecasting that the future close price equals the cutoff close:
        pred_close = last_close

    Keeping the same dates, horizons, and output columns as the other models makes this
    baseline directly comparable with N-BEATS, PatchTST, Chronos, and Kronos.
    """
    rows = []

    for horizon in config.horizons:
        cutoff_dates = make_cutoff_dates(
            panel,
            config.input_size,
            horizon,
            config.test_step,
            config.max_test_dates,
        )
        print(f"Naive baseline horizon {horizon}: {len(cutoff_dates)} cutoff dates")

        for ticker, g in panel.groupby("Ticker"):
            g = g.sort_values("Date").reset_index(drop=True)

            for cutoff_date in cutoff_dates:
                idx_list = g.index[g["Date"] == cutoff_date].tolist()
                if not idx_list:
                    continue

                idx = idx_list[0]
                if idx + horizon >= len(g) or idx + 1 < config.input_size:
                    continue

                last_close = float(g.loc[idx, "close"])
                actual_close = float(g.loc[idx + horizon, "close"])
                target_date = pd.Timestamp(g.loc[idx + horizon, "Date"])

                rows.append({
                    "Date": target_date,
                    "Ticker": ticker,
                    "Model": "Naive-RandomWalk",
                    "Horizon": horizon,
                    "y_true": actual_close / last_close - 1.0,
                    "y_pred": 0.0,
                    "cutoff_date": cutoff_date,
                    "last_close": last_close,
                    "pred_close": last_close,
                    "actual_close": actual_close,
                })

    return pd.DataFrame(rows)


# ── Classical statistical models ───────────────────────────────────────────────
# Uses statsforecast 1.7.4 (C implementation — ms per series, no numba needed).
# Install: pip install "statsforecast==1.7.4"

def run_classical_models(
    panel: pd.DataFrame,
    config: BenchmarkConfig,
    models_to_run: tuple[str, ...] = ("AutoARIMA", "AutoETS", "AutoTheta"),
) -> pd.DataFrame:
    """Walk-forward evaluation of classical statistical forecasting models.

    Uses statsforecast 1.7.4 (Nixtla) — C-optimised implementations,
    no numba/llvmlite dependency, parallelised across all CPU cores.

      - AutoARIMA  : automatic ARIMA order selection (AIC-based)
      - AutoETS    : automatic Exponential Smoothing / Holt-Winters
      - AutoTheta  : Theta decomposition method

    Returns the same column schema as all other model runners.
    """
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
    except ImportError as exc:
        raise ImportError(
            'statsforecast 1.7.4 required. '
            'Run: pip install "statsforecast==1.7.4"'
        ) from exc

    _registry = {
        "AutoARIMA": AutoARIMA(season_length=5, approximation=True, max_p=1, max_q=1, max_d=1),
        "AutoETS":   AutoETS(season_length=5),
        "AutoTheta": AutoTheta(season_length=5),
    }
    unknown = [m for m in models_to_run if m not in _registry]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Choose from {sorted(_registry)}")

    model_objects = [_registry[m] for m in models_to_run]
    sf_df = to_neuralforecast_df(panel)   # unique_id / ds / y — same format
    all_predictions: list = []

    max_horizon = max(config.horizons)
    cutoff_dates = make_cutoff_dates(
        panel, config.input_size, max_horizon, config.test_step, config.max_test_dates
    )
    print(f"Classical models: {len(cutoff_dates)} cutoffs  horizons={config.horizons}  models={list(models_to_run)}")

    for cutoff_date in cutoff_dates:
        train_df = sf_df[sf_df["ds"] <= cutoff_date].copy()
        # Use only last INPUT_SIZE rows per ticker — speeds up classical models significantly
        train_df = (train_df.sort_values("ds")
                    .groupby("unique_id", group_keys=False)
                    .tail(config.input_size))

        sf = StatsForecast(
            models=model_objects,
            freq=config.freq,
            n_jobs=-1,
        )
        fcst = sf.forecast(df=train_df, h=max_horizon).reset_index()
        fcst = fcst.sort_values(["unique_id", "ds"])
        fcst["step"] = fcst.groupby("unique_id").cumcount() + 1
        print(f"  cutoff={cutoff_date.date()}  rows={len(fcst[fcst['step']==1])}")

        for horizon in config.horizons:
            fcst_h = fcst[fcst["step"] == horizon].copy()
            for _, row in fcst_h.iterrows():
                ticker = row["unique_id"]
                actual = _actual_close_at(panel, ticker, cutoff_date, horizon)
                if actual is None:
                    continue
                target_date, last_close, actual_close = actual
                for model_name in models_to_run:
                    if model_name not in fcst_h.columns:
                        continue
                    pred_close = float(row[model_name])
                    if not np.isfinite(pred_close) or pred_close <= 0:
                        continue
                    all_predictions.append({
                        "Date":         target_date,
                        "Ticker":       ticker,
                        "Model":        model_name,
                        "Horizon":      horizon,
                        "y_true":       actual_close / last_close - 1.0,
                        "y_pred":       pred_close   / last_close - 1.0,
                        "cutoff_date":  cutoff_date,
                        "last_close":   last_close,
                        "pred_close":   pred_close,
                        "actual_close": actual_close,
                    })

    return pd.DataFrame(all_predictions)


class ChronosZeroShotForecaster:
    def __init__(self, model_id: str = "amazon/chronos-bolt-tiny", device: str | None = None):
        import torch
        from chronos import BaseChronosPipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.torch = torch
        self.pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
        self.model_id = model_id
        self.device = device

    def predict_close_path(self, context: np.ndarray, prediction_length: int) -> np.ndarray:
        tensor = self.torch.tensor(context, dtype=self.torch.float32)
        forecast = self.pipeline.predict(tensor, prediction_length=prediction_length)
        # Expected shape: [batch, num_samples, prediction_length]
        if hasattr(forecast, "detach"):
            arr = forecast.detach().cpu().numpy()
        else:
            arr = np.asarray(forecast)
        if arr.ndim == 3:
            return np.median(arr[0], axis=0)
        if arr.ndim == 2:
            return np.median(arr, axis=0)
        return arr.reshape(-1)


def run_chronos_zero_shot(panel: pd.DataFrame, config: BenchmarkConfig, model_id: str) -> pd.DataFrame:
    forecaster = ChronosZeroShotForecaster(model_id=model_id)
    rows = []

    for horizon in config.horizons:
        cutoff_dates = make_cutoff_dates(panel, config.input_size, horizon, config.test_step, config.max_test_dates)
        print(f"Chronos horizon {horizon}: {len(cutoff_dates)} cutoff dates")

        for ticker, g in panel.groupby("Ticker"):
            g = g.sort_values("Date").reset_index(drop=True)
            for cutoff_date in cutoff_dates:
                idx_list = g.index[g["Date"] == cutoff_date].tolist()
                if not idx_list:
                    continue
                idx = idx_list[0]
                if idx + horizon >= len(g) or idx + 1 < config.input_size:
                    continue
                context = g.loc[idx - config.input_size + 1 : idx, "close"].astype(float).values
                pred_path = forecaster.predict_close_path(context, prediction_length=horizon)
                pred_close = float(pred_path[horizon - 1])
                last_close = float(g.loc[idx, "close"])
                actual_close = float(g.loc[idx + horizon, "close"])
                target_date = pd.Timestamp(g.loc[idx + horizon, "Date"])
                rows.append({
                    "Date": target_date,
                    "Ticker": ticker,
                    "Model": "Chronos-ZeroShot",
                    "Horizon": horizon,
                    "y_true": actual_close / last_close - 1.0,
                    "y_pred": pred_close / last_close - 1.0,
                    "cutoff_date": cutoff_date,
                    "last_close": last_close,
                    "pred_close": pred_close,
                    "actual_close": actual_close,
                })
    return pd.DataFrame(rows)


class KronosZeroShotForecaster:
    def __init__(self, kronos_repo: Path, model_id: str, tokenizer_id: str, max_context: int = 512):
        if not kronos_repo.exists():
            raise FileNotFoundError(f"Kronos repo not found: {kronos_repo}")
        # Support both common Kronos layouts:
        #   Models/Kronos/model/kronos.py
        #   Models/Kronos/model.py
        sys.path.insert(0, str(kronos_repo))
        sys.path.insert(0, str(kronos_repo / "model"))
        try:
            from model import Kronos, KronosTokenizer, KronosPredictor
        except Exception:
            try:
                from kronos import Kronos, KronosTokenizer, KronosPredictor
            except Exception as exc:
                raise ImportError(
                    "Could not import Kronos classes. Check that KRONOS_LOCAL_REPO points "
                    "to your local Kronos repo and that its dependencies are installed."
                ) from exc

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        model = Kronos.from_pretrained(model_id)
        self.predictor = KronosPredictor(model, tokenizer, max_context=max_context)
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id

    def predict_close_path(self, hist_df: pd.DataFrame, future_dates: pd.Series, prediction_length: int) -> np.ndarray:
        hist_df = hist_df.copy()
        if "amount" not in hist_df.columns:
            hist_df["amount"] = hist_df["close"].astype(float) * hist_df["volume"].astype(float)
        x_df = hist_df[["open", "high", "low", "close", "volume", "amount"]].copy()
        x_timestamp = pd.to_datetime(hist_df["Date"])
        y_timestamp = pd.to_datetime(future_dates)
        pred_df = self.predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=prediction_length,
            T=1.0,
            top_p=0.9,
            sample_count=1,
        )
        return pred_df["close"].astype(float).values


def run_kronos_zero_shot(panel: pd.DataFrame, config: BenchmarkConfig, kronos_repo: Path, model_id: str, tokenizer_id: str) -> pd.DataFrame:
    forecaster = KronosZeroShotForecaster(kronos_repo=kronos_repo, model_id=model_id, tokenizer_id=tokenizer_id, max_context=min(512, config.input_size))
    rows = []

    for horizon in config.horizons:
        cutoff_dates = make_cutoff_dates(panel, config.input_size, horizon, config.test_step, config.max_test_dates)
        print(f"Kronos horizon {horizon}: {len(cutoff_dates)} cutoff dates")

        for ticker, g in panel.groupby("Ticker"):
            g = g.sort_values("Date").reset_index(drop=True)
            for cutoff_date in cutoff_dates:
                idx_list = g.index[g["Date"] == cutoff_date].tolist()
                if not idx_list:
                    continue
                idx = idx_list[0]
                if idx + horizon >= len(g) or idx + 1 < config.input_size:
                    continue
                hist_df = g.loc[idx - config.input_size + 1 : idx].copy()
                future_dates = g.loc[idx + 1 : idx + horizon, "Date"]
                pred_path = forecaster.predict_close_path(hist_df, future_dates, prediction_length=horizon)
                pred_close = float(pred_path[horizon - 1])
                last_close = float(g.loc[idx, "close"])
                actual_close = float(g.loc[idx + horizon, "close"])
                target_date = pd.Timestamp(g.loc[idx + horizon, "Date"])
                rows.append({
                    "Date": target_date,
                    "Ticker": ticker,
                    "Model": "Kronos-ZeroShot",
                    "Horizon": horizon,
                    "y_true": actual_close / last_close - 1.0,
                    "y_pred": pred_close / last_close - 1.0,
                    "cutoff_date": cutoff_date,
                    "last_close": last_close,
                    "pred_close": pred_close,
                    "actual_close": actual_close,
                })
    return pd.DataFrame(rows)

def run_naive_baseline(panel: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    """
    Naive / Random Walk baseline.

    Since the target is future return:
    y_pred = 0 means the model predicts no price change.

    Equivalent close-price forecast:
    pred_close = last_close
    """
    rows = []

    for horizon in config.horizons:
        cutoff_dates = make_cutoff_dates(
            panel,
            config.input_size,
            horizon,
            config.test_step,
            config.max_test_dates,
        )

        print(f"Naive baseline horizon {horizon}: {len(cutoff_dates)} cutoff dates")

        for ticker, g in panel.groupby("Ticker"):
            g = g.sort_values("Date").reset_index(drop=True)

            for cutoff_date in cutoff_dates:
                idx_list = g.index[g["Date"] == cutoff_date].tolist()
                if not idx_list:
                    continue

                idx = idx_list[0]

                if idx + horizon >= len(g):
                    continue

                last_close = float(g.loc[idx, "close"])
                actual_close = float(g.loc[idx + horizon, "close"])
                target_date = pd.Timestamp(g.loc[idx + horizon, "Date"])

                rows.append({
                    "Date": target_date,
                    "Ticker": ticker,
                    "Model": "Naive-RandomWalk",
                    "Horizon": horizon,
                    "y_true": actual_close / last_close - 1.0,
                    "y_pred": 0.0,
                    "cutoff_date": cutoff_date,
                    "last_close": last_close,
                    "pred_close": last_close,
                    "actual_close": actual_close,
                })

    return pd.DataFrame(rows)

def run_naive_baseline(panel: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    """
    Naive / Random Walk baseline.

    Since the target is future return:
    y_pred = 0 means the model predicts no price change.

    Equivalent close-price forecast:
    pred_close = last_close
    """
    rows = []

    for horizon in config.horizons:
        cutoff_dates = make_cutoff_dates(
            panel,
            config.input_size,
            horizon,
            config.test_step,
            config.max_test_dates,
        )

        print(f"Naive baseline horizon {horizon}: {len(cutoff_dates)} cutoff dates")

        for ticker, g in panel.groupby("Ticker"):
            g = g.sort_values("Date").reset_index(drop=True)

            for cutoff_date in cutoff_dates:
                idx_list = g.index[g["Date"] == cutoff_date].tolist()
                if not idx_list:
                    continue

                idx = idx_list[0]

                if idx + horizon >= len(g):
                    continue

                last_close = float(g.loc[idx, "close"])
                actual_close = float(g.loc[idx + horizon, "close"])
                target_date = pd.Timestamp(g.loc[idx + horizon, "Date"])

                rows.append({
                    "Date": target_date,
                    "Ticker": ticker,
                    "Model": "Naive-RandomWalk",
                    "Horizon": horizon,
                    "y_true": actual_close / last_close - 1.0,
                    "y_pred": 0.0,
                    "cutoff_date": cutoff_date,
                    "last_close": last_close,
                    "pred_close": last_close,
                    "actual_close": actual_close,
                })

    return pd.DataFrame(rows)