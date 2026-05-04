# Signal engine — imported lazily so a missing dependency never breaks
# the original notebooks (01-04) that only import src.config / src.data_utils.
try:
    from .data_feeds          import fetch_multi_timeframe, fetch_ohlcv, LiveDataFeed
    from .feature_engineering import compute_features, build_multi_tf_snapshot
    from .signal_engine       import SignalEngine, Signal
    from .risk_manager        import RiskManager, TradeSetup
    from .backtester          import Backtester
    from .signal_logger       import SignalLogger
except Exception:
    pass

try:
    from .new_models import run_classical_models, run_naive_baseline, BenchmarkConfig
except Exception:
    pass
