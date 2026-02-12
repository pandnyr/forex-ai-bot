"""
Backtest Data Loader - Load historical data from MT5 or CSV files
Supports Dukascopy tick data import
"""
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Try to import MT5 - may not be available on all platforms
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not available - using CSV data only")

TIMEFRAME_MAP = {
    "M1": {"mt5": 1, "minutes": 1},
    "M5": {"mt5": 5, "minutes": 5},
    "M15": {"mt5": 15, "minutes": 15},
    "M30": {"mt5": 30, "minutes": 30},
    "H1": {"mt5": 16385, "minutes": 60},
    "H4": {"mt5": 16388, "minutes": 240},
    "D1": {"mt5": 16408, "minutes": 1440},
}


class DataLoader:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "..", "data", "historical")
        os.makedirs(self.data_dir, exist_ok=True)

    def load_mt5(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Load data from MT5 terminal."""
        if not MT5_AVAILABLE:
            raise RuntimeError("MT5 not available on this platform")

        tf_info = TIMEFRAME_MAP.get(timeframe)
        if not tf_info:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        rates = mt5.copy_rates_range(symbol, tf_info["mt5"], start, end)
        if rates is None or len(rates) == 0:
            raise ValueError(f"No data returned for {symbol} {timeframe}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        # Cache to CSV
        cache_path = self._cache_path(symbol, timeframe, start, end)
        df.to_csv(cache_path)
        logger.info(f"Loaded {len(df)} bars from MT5: {symbol} {timeframe}")
        return df[["open", "high", "low", "close", "volume"]]

    def load_csv(self, filepath: str, date_column: str = "time") -> pd.DataFrame:
        """Load data from CSV file (Dukascopy, HistData, etc.)."""
        df = pd.read_csv(filepath, parse_dates=[date_column])
        if date_column != "time":
            df.rename(columns={date_column: "time"}, inplace=True)

        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)

        # Normalize column names
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if "open" in lower:
                col_map[col] = "open"
            elif "high" in lower:
                col_map[col] = "high"
            elif "low" in lower:
                col_map[col] = "low"
            elif "close" in lower:
                col_map[col] = "close"
            elif "vol" in lower:
                col_map[col] = "volume"
        df.rename(columns=col_map, inplace=True)

        required = ["open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        if "volume" not in df.columns:
            df["volume"] = 0

        logger.info(f"Loaded {len(df)} bars from CSV: {filepath}")
        return df[["open", "high", "low", "close", "volume"]]

    def load(self, symbol: str, timeframe: str, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """Smart loader: try cache first, then MT5, then error."""
        # Try cache
        if start and end:
            cache_path = self._cache_path(symbol, timeframe, start, end)
            if os.path.exists(cache_path):
                logger.info(f"Loading from cache: {cache_path}")
                df = pd.read_csv(cache_path, parse_dates=["time"], index_col="time")
                return df

        # Try MT5
        if MT5_AVAILABLE and start and end:
            try:
                return self.load_mt5(symbol, timeframe, start, end)
            except Exception as e:
                logger.warning(f"MT5 load failed: {e}")

        # Try to find any CSV in data_dir
        pattern = f"{symbol}_{timeframe}".lower()
        for f in os.listdir(self.data_dir):
            if pattern in f.lower() and f.endswith(".csv"):
                return self.load_csv(os.path.join(self.data_dir, f))

        raise FileNotFoundError(f"No data found for {symbol} {timeframe}")

    def resample(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample lower timeframe to higher timeframe."""
        minutes = TIMEFRAME_MAP.get(target_tf, {}).get("minutes")
        if not minutes:
            raise ValueError(f"Invalid target timeframe: {target_tf}")

        rule = f"{minutes}min" if minutes < 1440 else "1D"
        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        return resampled

    def prepare_multi_tf(self, symbol: str, start: datetime, end: datetime,
                         base_tf: str = "M5") -> dict:
        """Load base timeframe and resample to create multi-TF dataset."""
        base = self.load(symbol, base_tf, start, end)

        result = {base_tf: base}

        # Generate higher timeframes by resampling
        higher_tfs = {"M15": 15, "H1": 60, "H4": 240}
        for tf, mins in higher_tfs.items():
            base_mins = TIMEFRAME_MAP[base_tf]["minutes"]
            if mins > base_mins:
                result[tf] = self.resample(base, tf)

        return result

    def _cache_path(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> str:
        s = start.strftime("%Y%m%d")
        e = end.strftime("%Y%m%d")
        return os.path.join(self.data_dir, f"{symbol}_{timeframe}_{s}_{e}.csv")

    def add_spread_simulation(self, df: pd.DataFrame, avg_spread_pips: float = 1.0,
                               symbol_digits: int = 5) -> pd.DataFrame:
        """Add simulated spread column for more realistic backtesting."""
        pip_size = 0.0001 if symbol_digits == 5 else 0.01
        half_spread = (avg_spread_pips * pip_size) / 2

        df = df.copy()
        # Add random spread variation (80% to 150% of average)
        np.random.seed(42)
        spread_mult = np.random.uniform(0.8, 1.5, len(df))
        df["spread"] = avg_spread_pips * spread_mult
        df["ask"] = df["close"] + half_spread * spread_mult
        df["bid"] = df["close"] - half_spread * spread_mult
        return df
