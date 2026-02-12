"""
Mean Reversion Strategy - Asian Session Bollinger Band + RSI Extremes
Designed for ranging (low ADX) market conditions.

Entry Logic:
  - Long:  RSI < 30 (oversold) + price below lower BB(20,2) + ADX < 20
  - Short: RSI > 70 (overbought) + price above upper BB(20,2) + ADX < 20
  - Only trade during Asian session (00:00-06:00 UTC)
  - Prefer USDJPY in Asian session

SL: 1x ATR(14) converted to pips
TP: Distance to Bollinger middle band (converted to pips)
"""
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def _points_to_pips(points: float, digits: int = 5) -> float:
    """Convert price difference to pips."""
    if digits == 5:
        return points / 0.00010
    elif digits == 3:
        return points / 0.010
    elif digits == 4:
        return points / 0.0001
    elif digits == 2:
        return points / 0.01
    return points


def _get_digits(symbol: str) -> int:
    """Infer digit precision from symbol name."""
    sym = symbol.upper()
    if "JPY" in sym:
        return 3
    return 5


class MeanReversionStrategy:
    """Asian Session Mean Reversion Strategy - BB + RSI in Ranging Markets."""

    def __init__(self, config: dict):
        """
        Initialize with config dictionary.

        Expected config keys:
            MEAN_REV_RSI_OVERSOLD (float): RSI oversold threshold, default 30
            MEAN_REV_RSI_OVERBOUGHT (float): RSI overbought threshold, default 70
            MEAN_REV_BB_PERIOD (int): Bollinger Band period, default 20
            MEAN_REV_BB_STD (float): Bollinger Band std dev, default 2.0
            MEAN_REV_ATR_MULT_SL (float): SL multiplier for ATR, default 1.0
            SCALP_RSI_PERIOD (int): RSI period, default 14
            SCALP_ATR_PERIOD (int): ATR period, default 14
            TREND_ADX_PERIOD (int): ADX period, default 14
            ASIAN_START (int): Asian session start hour UTC, default 0
            ASIAN_END (int): Asian session end hour UTC, default 6
        """
        self._config = config
        self._rsi_oversold = config.get("MEAN_REV_RSI_OVERSOLD", 30)
        self._rsi_overbought = config.get("MEAN_REV_RSI_OVERBOUGHT", 70)
        self._bb_period = config.get("MEAN_REV_BB_PERIOD", 20)
        self._bb_std = config.get("MEAN_REV_BB_STD", 2.0)
        self._atr_mult_sl = config.get("MEAN_REV_ATR_MULT_SL", 1.0)
        self._rsi_period = config.get("SCALP_RSI_PERIOD", 14)
        self._atr_period = config.get("SCALP_ATR_PERIOD", 14)
        self._adx_period = config.get("TREND_ADX_PERIOD", 14)
        self._adx_max = 20  # Only trade in ranging markets (ADX < 20)
        self._asian_start = config.get("ASIAN_START", 0)
        self._asian_end = 6  # Strict Asian session: 00:00-06:00 UTC

    @property
    def name(self) -> str:
        return "mean_reversion"

    # ------------------------------------------------------------------
    # Session filter
    # ------------------------------------------------------------------

    def _is_asian_session(self, df: pd.DataFrame) -> bool:
        """Check if the latest bar falls within the Asian session (00:00-06:00 UTC)."""
        if len(df) == 0:
            return False

        last_time = df.index[-1]
        if hasattr(last_time, "hour"):
            hour = last_time.hour
            return self._asian_start <= hour < self._asian_end
        return False

    # ------------------------------------------------------------------
    # Market regime filter
    # ------------------------------------------------------------------

    def _is_ranging_market(self, df: pd.DataFrame) -> tuple[bool, float]:
        """Check if market is ranging (ADX < 20).

        Returns:
            (is_ranging: bool, adx_value: float)
        """
        if len(df) < self._adx_period + 5:
            return False, 0.0

        adx_df = df.ta.adx(length=self._adx_period)
        if adx_df is None:
            return False, 0.0

        adx_col = f"ADX_{self._adx_period}"
        if adx_col not in adx_df.columns:
            return False, 0.0

        last_adx = adx_df[adx_col].iloc[-1]
        if np.isnan(last_adx):
            return False, 0.0

        return last_adx < self._adx_max, last_adx

    # ------------------------------------------------------------------
    # Signal detection
    # ------------------------------------------------------------------

    def _check_long_setup(self, df: pd.DataFrame) -> tuple[bool, dict]:
        """Check for long (oversold bounce) setup.

        Conditions:
          - RSI < 30
          - Price below lower Bollinger Band
          - ADX < 20

        Returns:
            (setup_valid: bool, details: dict)
        """
        if len(df) < max(self._bb_period, self._rsi_period) + 5:
            return False, {}

        rsi = df.ta.rsi(length=self._rsi_period)
        bbands = df.ta.bbands(length=self._bb_period, std=self._bb_std)

        if rsi is None or bbands is None:
            return False, {}

        last_rsi = rsi.iloc[-1]
        last_close = df["close"].iloc[-1]

        # Bollinger Band column names from pandas_ta
        bbl_col = f"BBL_{self._bb_period}_{self._bb_std}"
        bbm_col = f"BBM_{self._bb_period}_{self._bb_std}"

        if bbl_col not in bbands.columns or bbm_col not in bbands.columns:
            return False, {}

        last_bbl = bbands[bbl_col].iloc[-1]
        last_bbm = bbands[bbm_col].iloc[-1]

        if np.isnan(last_rsi) or np.isnan(last_bbl) or np.isnan(last_bbm):
            return False, {}

        # Check oversold conditions
        rsi_oversold = last_rsi < self._rsi_oversold
        below_lower_bb = last_close < last_bbl

        if rsi_oversold and below_lower_bb:
            return True, {
                "rsi": round(last_rsi, 2),
                "bb_lower": last_bbl,
                "bb_middle": last_bbm,
                "close": last_close,
            }

        return False, {}

    def _check_short_setup(self, df: pd.DataFrame) -> tuple[bool, dict]:
        """Check for short (overbought reversal) setup.

        Conditions:
          - RSI > 70
          - Price above upper Bollinger Band
          - ADX < 20
        """
        if len(df) < max(self._bb_period, self._rsi_period) + 5:
            return False, {}

        rsi = df.ta.rsi(length=self._rsi_period)
        bbands = df.ta.bbands(length=self._bb_period, std=self._bb_std)

        if rsi is None or bbands is None:
            return False, {}

        last_rsi = rsi.iloc[-1]
        last_close = df["close"].iloc[-1]

        bbu_col = f"BBU_{self._bb_period}_{self._bb_std}"
        bbm_col = f"BBM_{self._bb_period}_{self._bb_std}"

        if bbu_col not in bbands.columns or bbm_col not in bbands.columns:
            return False, {}

        last_bbu = bbands[bbu_col].iloc[-1]
        last_bbm = bbands[bbm_col].iloc[-1]

        if np.isnan(last_rsi) or np.isnan(last_bbu) or np.isnan(last_bbm):
            return False, {}

        rsi_overbought = last_rsi > self._rsi_overbought
        above_upper_bb = last_close > last_bbu

        if rsi_overbought and above_upper_bb:
            return True, {
                "rsi": round(last_rsi, 2),
                "bb_upper": last_bbu,
                "bb_middle": last_bbm,
                "close": last_close,
            }

        return False, {}

    # ------------------------------------------------------------------
    # SL/TP calculation
    # ------------------------------------------------------------------

    def _compute_sl_pips(self, df: pd.DataFrame, symbol: str) -> float:
        """Compute SL in pips: 1x ATR(14)."""
        digits = _get_digits(symbol)

        atr = df.ta.atr(length=self._atr_period)
        if atr is None or len(atr) == 0:
            return 12.0

        last_atr = atr.iloc[-1]
        if np.isnan(last_atr):
            return 12.0

        sl_price = last_atr * self._atr_mult_sl
        sl_pips = _points_to_pips(sl_price, digits)

        # Clamp: mean reversion has tighter stops
        return round(max(5.0, min(sl_pips, 40.0)), 1)

    def _compute_tp_pips(self, current_close: float, bb_middle: float,
                         direction: str, symbol: str) -> float:
        """Compute TP as distance to Bollinger middle band in pips.

        For longs:  TP = BB_middle - current_close
        For shorts: TP = current_close - BB_middle
        """
        digits = _get_digits(symbol)

        if direction == "buy":
            tp_price = bb_middle - current_close
        else:
            tp_price = current_close - bb_middle

        tp_pips = _points_to_pips(abs(tp_price), digits)

        # Minimum TP of 5 pips
        return round(max(5.0, tp_pips), 1)

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(self, direction: str, details: dict,
                            adx_value: float, df: pd.DataFrame,
                            symbol: str) -> float:
        """Compute confidence score (0-100).

        Scoring:
          - RSI extremity:             +25 (more extreme = more confident)
          - BB penetration depth:      +20 (deeper below/above = stronger signal)
          - ADX confirms ranging:      +20 (lower ADX = more ranging = better)
          - Volume confirmation:       +15 (volume below average = consolidation)
          - Asian session premium:     +10 (USDJPY gets bonus)
          - Recent BB touch history:   +10 (if price bounced off BB before)
        """
        score = 0.0

        rsi_val = details.get("rsi", 50.0)

        # RSI extremity
        if direction == "buy":
            rsi_extreme = max(0.0, (self._rsi_oversold - rsi_val) / self._rsi_oversold) * 25.0
        else:
            rsi_extreme = max(0.0, (rsi_val - self._rsi_overbought) / (100.0 - self._rsi_overbought)) * 25.0
        score += rsi_extreme

        # BB penetration depth
        close = details.get("close", 0.0)
        if direction == "buy":
            bb_edge = details.get("bb_lower", close)
            if bb_edge > 0:
                penetration = max(0.0, (bb_edge - close) / bb_edge) * 10000  # In pips-like scale
                score += min(penetration / 5.0, 20.0)
        else:
            bb_edge = details.get("bb_upper", close)
            if bb_edge > 0:
                penetration = max(0.0, (close - bb_edge) / bb_edge) * 10000
                score += min(penetration / 5.0, 20.0)

        # ADX ranging confirmation
        if adx_value > 0:
            # Lower ADX = better for mean reversion (max score at ADX ~10)
            adx_score = max(0.0, (self._adx_max - adx_value) / self._adx_max) * 20.0
            score += adx_score

        # Volume below average = consolidation (good for mean reversion)
        if "volume" in df.columns and len(df) >= 20:
            recent_vol = df["volume"].iloc[-1]
            avg_vol = df["volume"].iloc[-20:].mean()
            if avg_vol > 0 and not np.isnan(recent_vol) and not np.isnan(avg_vol):
                if recent_vol < avg_vol:
                    vol_ratio = recent_vol / avg_vol
                    score += (1.0 - vol_ratio) * 15.0

        # USDJPY bonus during Asian session
        if "JPY" in symbol.upper():
            score += 10.0

        # Recent BB touch (price touched BB and reversed within last 3 bars)
        bbands = df.ta.bbands(length=self._bb_period, std=self._bb_std)
        if bbands is not None and len(df) >= 3:
            bbl_col = f"BBL_{self._bb_period}_{self._bb_std}"
            bbu_col = f"BBU_{self._bb_period}_{self._bb_std}"
            if bbl_col in bbands.columns and bbu_col in bbands.columns:
                for i in range(-3, -1):
                    if abs(i) > len(df):
                        continue
                    bar_low = df["low"].iloc[i]
                    bar_high = df["high"].iloc[i]
                    if direction == "buy":
                        bb_val = bbands[bbl_col].iloc[i]
                        if not np.isnan(bb_val) and bar_low <= bb_val:
                            score += 10.0
                            break
                    else:
                        bb_val = bbands[bbu_col].iloc[i]
                        if not np.isnan(bb_val) and bar_high >= bb_val:
                            score += 10.0
                            break

        return round(min(score, 100.0), 1)

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(self, symbol: str, bars_dict: dict) -> dict | None:
        """Analyze market data and return a Signal dict or None.

        Uses H1 as the primary timeframe. Falls back to M30 from the
        M15 data if H1 data is too short.

        Args:
            symbol: e.g. "USDJPY"
            bars_dict: {"M5": DataFrame, "M15": DataFrame, "H1": DataFrame, "H4": DataFrame}

        Returns:
            Signal dict or None.
        """
        h1 = bars_dict.get("H1")

        if h1 is None:
            logger.debug(f"[{self.name}] Missing H1 data for {symbol}")
            return None

        min_bars = max(self._bb_period, self._rsi_period, self._adx_period) + 10
        if len(h1) < min_bars:
            logger.debug(f"[{self.name}] Insufficient H1 bars for {symbol} "
                         f"(have {len(h1)}, need {min_bars})")
            return None

        # Step 1: Session filter - only trade during Asian session
        if not self._is_asian_session(h1):
            logger.debug(f"[{self.name}] {symbol} outside Asian session")
            return None

        # Step 2: Market regime filter - ADX must be < 20
        is_ranging, adx_value = self._is_ranging_market(h1)
        if not is_ranging:
            logger.debug(f"[{self.name}] {symbol} not ranging (ADX={adx_value:.1f})")
            return None

        # Step 3: Check for setups
        long_valid, long_details = self._check_long_setup(h1)
        short_valid, short_details = self._check_short_setup(h1)

        direction = None
        details = {}

        if long_valid:
            direction = "buy"
            details = long_details
        elif short_valid:
            direction = "sell"
            details = short_details
        else:
            logger.debug(f"[{self.name}] {symbol} no mean reversion setup found")
            return None

        # Step 4: Compute SL
        sl_pips = self._compute_sl_pips(h1, symbol)

        # Step 5: Compute TP (distance to BB middle band)
        bb_middle = details.get("bb_middle", 0.0)
        current_close = details.get("close", 0.0)
        tp_pips = self._compute_tp_pips(current_close, bb_middle, direction, symbol)

        # Ensure TP > SL for favorable R:R (at least 1:1)
        if tp_pips < sl_pips:
            logger.debug(f"[{self.name}] {symbol} TP ({tp_pips}) < SL ({sl_pips}), "
                         f"unfavorable R:R, skipping")
            return None

        # Step 6: Confidence scoring
        confidence = self._compute_confidence(direction, details, adx_value, h1, symbol)

        rsi_val = details.get("rsi", 0.0)
        reasoning = (
            f"Asian session mean reversion: "
            f"RSI={rsi_val} ({'oversold' if direction == 'buy' else 'overbought'}), "
            f"price {'below lower' if direction == 'buy' else 'above upper'} BB({self._bb_period},{self._bb_std}), "
            f"ADX={adx_value:.1f} (ranging market), "
            f"SL={sl_pips} pips (ATR*{self._atr_mult_sl}), "
            f"TP={tp_pips} pips (to BB middle band)"
        )

        logger.info(f"[{self.name}] SIGNAL: {direction.upper()} {symbol} | "
                     f"confidence={confidence} | RSI={rsi_val}")

        return {
            "symbol": symbol,
            "direction": direction,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "confidence": confidence,
            "strategy": self.name,
            "reasoning": reasoning,
        }
