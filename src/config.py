"""
Forex AI Trading Bot - Configuration
Brain + Muscles Pattern (Sharbel approach)
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# API Keys
# ============================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")
GOOGLE_SHEETS_CREDS = os.getenv("GOOGLE_SHEETS_CREDS", "credentials.json")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Trading Journal")

# ============================================================
# MT5 Configuration
# ============================================================
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", "")  # Path to terminal64.exe

# ============================================================
# Exchange & Symbols
# ============================================================
EXCHANGE = "mt5"
MT5_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
MT5_TIMEFRAMES = {"scalp": "M5", "intraday": "M15", "day": "H1", "swing": "H4", "daily": "D1"}
MT5_MAGIC_NUMBER = 234000  # Unique identifier for our bot's orders

# ============================================================
# Risk Management
# ============================================================
RISK_PER_TRADE = 1.0          # % risk per trade
MAX_DAILY_TRADES = 8          # Max trades per day
MAX_DAILY_LOSS_PCT = 3.0      # % max daily loss
MAX_TOTAL_DD_PCT = 10.0       # % max total drawdown from peak
MAX_CONCURRENT_POSITIONS = 3  # Max open positions at once
MAX_PER_PAIR_POSITIONS = 1    # Max positions per pair
MIN_ACCOUNT_BALANCE = 50.0    # Minimum balance to keep trading ($)
MAX_DAILY_PROFIT_PCT = 5.0    # Stop trading after 5% daily profit (overtrading prevention)

# Anti-martingale settings
CONSECUTIVE_LOSS_REDUCE = 2   # After N consecutive losses, reduce lot by 50%
CONSECUTIVE_WIN_INCREASE = 3  # After N consecutive wins, increase lot by 25%

# ============================================================
# Compound & Pyramiding
# ============================================================
COMPOUND_ENABLED = True       # Recalculate lot size after each trade based on new balance
PYRAMIDING_ENABLED = True     # Allow adding to winning positions
PYRAMID_MAX_ADDS = 2          # Max additions to a winning position
PYRAMID_SCALE_FACTOR = 0.5    # Each addition is 50% of original size
PYRAMID_MIN_PROFIT_PIPS = 15  # Min profit before first pyramid add

# ============================================================
# Session Configuration (UTC)
# ============================================================
LONDON_START, LONDON_END = 7, 16
NY_START, NY_END = 13, 21
ASIAN_START, ASIAN_END = 0, 8
OVERLAP_START, OVERLAP_END = 13, 16  # London-NY overlap (best volatility)

# ============================================================
# AI Model Configuration (Brain + Muscles Pattern)
# ============================================================
BRAIN_MODEL = "anthropic/claude-sonnet-4-20250514"    # Critical decisions (via OpenRouter)
MUSCLE_MODEL = "anthropic/claude-3.5-haiku-20241022"  # Monitoring, routine (via OpenRouter)
AI_MAX_TOKENS = 1024
AI_TEMPERATURE = 0.3

# ============================================================
# Strategy Parameters
# ============================================================
# Scalp Momentum (Main strategy)
SCALP_EMA_FAST = 20
SCALP_EMA_SLOW = 50
SCALP_RSI_PERIOD = 14
SCALP_RSI_THRESHOLD = 50
SCALP_ATR_PERIOD = 14
SCALP_SL_ATR_MULT = 1.5      # SL = 1.5x ATR
SCALP_TP_RR = 2.0             # TP = 2x SL (1:2 R:R)
SCALP_MAX_SPREAD_PIPS = 1.5   # Don't trade if spread > 1.5 pips
SCALP_MIN_AI_CONFIDENCE = 65  # Min AI confidence score to take trade

# Trend Following
TREND_EMA_FAST = 20
TREND_EMA_SLOW = 50
TREND_ADX_PERIOD = 14
TREND_ADX_THRESHOLD = 25
TREND_ATR_MULT_SL = 1.5
TREND_TP_RR = 2.5

# Mean Reversion
MEAN_REV_RSI_OVERSOLD = 30
MEAN_REV_RSI_OVERBOUGHT = 70
MEAN_REV_BB_PERIOD = 20
MEAN_REV_BB_STD = 2.0
MEAN_REV_ATR_MULT_SL = 1.0

# Breakout
BREAKOUT_LOOKBACK_BARS = 12   # Pre-London range bars (4:00-7:00 = 12 bars on M15)
BREAKOUT_TP_MULT = 1.5        # TP = 1.5x range width
BREAKOUT_MAX_RANGE_PIPS = 40  # Skip if range > 40 pips (too wide)
BREAKOUT_MIN_RANGE_PIPS = 10  # Skip if range < 10 pips (too narrow)

# ============================================================
# Backtest Parameters
# ============================================================
BACKTEST_INITIAL_BALANCE = 10000.0
BACKTEST_COMMISSION_PER_LOT = 7.0  # $7 per lot round trip
BACKTEST_SLIPPAGE_PIPS = 0.5
BACKTEST_MIN_SHARPE = 1.0
BACKTEST_MIN_PROFIT_FACTOR = 1.5
BACKTEST_MIN_WIN_RATE = 0.50
BACKTEST_MIN_TRADES = 200
BACKTEST_MAX_DD_PCT = 20.0

# Walk-Forward
WFO_TRAIN_MONTHS = 6
WFO_TEST_MONTHS = 3
WFO_STEP_MONTHS = 3

# ============================================================
# Journal
# ============================================================
JOURNAL_COLUMNS = [
    "trade_id", "datetime_open", "datetime_close", "symbol", "direction",
    "entry_price", "exit_price", "sl", "tp", "lot_size",
    "pnl_usd", "pnl_pips", "r_multiple", "strategy", "session",
    "regime", "ai_confidence", "duration_minutes", "commission",
    "ai_analysis", "learner_notes"
]

# ============================================================
# Bot Loop
# ============================================================
LOOP_INTERVAL_SECONDS = 30    # Main loop sleep
LEARNER_INTERVAL_TRADES = 20  # Run cumulative learning every N trades

# ============================================================
# Logging
# ============================================================
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "data")
