from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "new_models"
PLOTS_DIR = RESULTS_DIR / "plots"

for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "GC=F", "META", "NVDA", "TSLA", "AMD", "SI=F",
    "INTC", "QCOM", "AVGO", "ADBE", "CL=F", "CRM", "ORCL", "CSCO", "IBM", "BZ=F",
    "TXN", "MU", "KLAC", "LRCX", "NG=F", "PANW", "CRWD", "NET", "PLTR", "HG=F",
    "SNOW", "JPM", "BAC", "GS", "PL=F", "MS", "V", "MA", "AXP", "ZC=F",
    "WFC", "C", "BLK", "SPGI", "ZS=F", "MCO", "PYPL", "USB", "PNC", "ZW=F",
    "TFC", "ICE", "CME", "BX", "EURUSD=X", "KKR", "JNJ", "UNH", "LLY", "USDJPY=X",
    "PFE", "ABBV", "AMGN", "GILD", "GBPUSD=X", "MRK", "BMY", "TMO", "ABT", "AUDUSD=X",
    "MDT", "BSX", "SYK", "ISRG", "USDCHF=X", "WMT", "PG", "KO", "PEP", "USDCAD=X",
    "COST", "MDLZ", "CL", "MO", "NZDUSD=X", "PM", "EL", "MCD", "SBUX", "EURGBP=X",
    "NKE", "HD", "LOW", "TGT", "EURJPY=X", "BKNG", "CMG", "ORLY", "TJX", "GBPJPY=X",
    "XOM", "CVX", "COP", "EOG", "AUDJPY=X", "SLB", "PSX", "VLO", "MPC", "CADJPY=X",
    "OXY", "HAL", "HON", "CAT", "EURCHF=X", "DE", "BA", "GE", "MMM", "EURAUD=X",
    "UPS", "FDX", "LMT", "RTX", "EURCAD=X", "DIS", "NFLX", "CMCSA", "T", "GBPCHF=X",
    "VZ", "SPOT", "SNAP", "ZM", "GBPAUD=X", "LIN", "APD", "SHW", "FCX", "GBPCAD=X",
    "NEM", "ECL", "AMT", "PLD", "AUDCAD=X", "EQIX", "PSA", "WELL", "O", "CHFJPY=X",
    "UBER", "ABNB", "ETSY", "EBAY", "DELL", "HPQ", "WM", "RSG", "INTU", "NOW",
]

START_DATE = "2019-01-01"
END_DATE = None  # None means today in yfinance

# Low-compute settings. Increase later only after the pipeline works.
HORIZONS = [1, 5]
INPUT_SIZE = 252
TEST_STEP = 21
MAX_TEST_DATES = 18
MAX_STEPS = 100
FREQ = "B"
MIN_ASSETS_FOR_RANKIC = 20
RANDOM_SEED = 42

# Foundation model settings.
CHRONOS_MODEL_ID = "amazon/chronos-bolt-tiny"
KRONOS_MODEL_ID = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_LOCAL_REPO = PROJECT_ROOT / "models" / "Kronos"
