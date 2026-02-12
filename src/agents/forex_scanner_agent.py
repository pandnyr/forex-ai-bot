"""
Forex Scanner Agent - Market scanning and opportunity detection
"""
import logging
from ..config import MT5_SYMBOLS, MT5_TIMEFRAMES
from .. import nice_funcs_mt5 as mt5f

logger = logging.getLogger(__name__)


class ForexScannerAgent:
    def scan_all_pairs(self, symbols: list = None) -> dict:
        """
        Scan all pairs and return market data for each.
        Returns: {"EURUSD": {"M5": df, "M15": df, "H1": df, "H4": df}, ...}
        """
        symbols = symbols or MT5_SYMBOLS
        result = {}

        for symbol in symbols:
            bars = {}
            for tf_name, tf_code in MT5_TIMEFRAMES.items():
                count = {"scalp": 200, "intraday": 100, "day": 100, "swing": 50, "daily": 30}
                df = mt5f.get_bars(symbol, tf_code, count.get(tf_name, 100))
                if not df.empty:
                    bars[tf_code] = df

            if bars:
                result[symbol] = bars
                spread = mt5f.get_spread(symbol)
                logger.info(f"Scanned {symbol}: {len(bars)} timeframes, spread={spread:.1f} pips")

        return result

    def get_market_overview(self, symbols: list = None) -> list:
        """Get a quick overview of all pairs: trend, volatility, spread."""
        import pandas_ta as ta
        symbols = symbols or MT5_SYMBOLS
        overview = []

        for symbol in symbols:
            h1 = mt5f.get_bars(symbol, "H1", 100)
            if h1.empty:
                continue

            # Calculate indicators
            ema20 = ta.ema(h1["close"], length=20)
            ema50 = ta.ema(h1["close"], length=50)
            adx_df = ta.adx(h1["high"], h1["low"], h1["close"], length=14)
            atr = ta.atr(h1["high"], h1["low"], h1["close"], length=14)
            rsi = ta.rsi(h1["close"], length=14)

            if ema20 is None or ema50 is None:
                continue

            last_close = h1["close"].iloc[-1]
            trend = "bullish" if ema20.iloc[-1] > ema50.iloc[-1] else "bearish"

            adx_val = adx_df[f"ADX_14"].iloc[-1] if adx_df is not None else 0
            regime = "trending" if adx_val > 25 else ("ranging" if adx_val < 20 else "transitional")

            overview.append({
                "symbol": symbol,
                "price": last_close,
                "trend": trend,
                "regime": regime,
                "adx": round(adx_val, 1),
                "rsi": round(rsi.iloc[-1], 1) if rsi is not None else 50,
                "atr": round(atr.iloc[-1], 5) if atr is not None else 0,
                "spread": mt5f.get_spread(symbol),
                "ema20": round(ema20.iloc[-1], 5),
                "ema50": round(ema50.iloc[-1], 5),
            })

        return overview
