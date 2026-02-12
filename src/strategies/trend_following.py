"""
Trend Following Strategy - London Session EMA Crossover + Pullback
Designed for H1 entries with H4 trend confirmation.

Entry Logic:
  - H4 trend: EMA20/EMA50 crossover establishes trend direction
  - H1 entry: EMA20/EMA50 aligned + ADX > 25 (strong trend)
  - Pullback to EMA20: price was > 5 pips above EMA20, now within 3 pips
  - Only trade during London session (07:00-10:00 UTC)

SL: 1.5x ATR(14) on H1
TP: 2.5x SL
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


def _pips_to_price(pips: float, digits: int = 5) -> float:
    """Convert pips to price distance."""
    if digits == 5:
        return pips * 0.00010
    elif digits == 3:
        return pips * 0.010
    elif digits == 4:
        return pips * 0.0001
    elif digits == 2:
        return pips * 0.01
    return pips


def _get_digits(symbol: str) -> int:
    """Infer digit precision from symbol name."""
    sym = symbol.upper()
    if "JPY" in sym:
        return 3
    return 5


class TrendFollowingStrategy:
    """London Trend Following Strategy - H1 EMA Crossover + Pullback."""

    def __init__(self, config: dict):
        """
        Initialize with config dictionary.

        Expected config keys:
            TREND_EMA_FAST (int): Fast EMA period, default 20
            TREND_EMA_SLOW (int): Slow EMA period, default 50
            TREND_ADX_PERIOD (int): ADX period, default 14
            TREND_ADX_THRESHOLD (float): ADX minimum for trend, default 25
            TREND_ATR_MULT_SL (float): SL multiplier for ATR, default 1.5
            TREND_TP_RR (float): TP as multiple of SL, default 2.5
            LONDON_START (int): London session start hour UTC, default 7
            LONDON_END (int): London open window end hour UTC, default 10
        """
        self._config = config
        self._ema_fast = config.get("TREND_EMA_FAST", 20)
        self._ema_slow = config.get("TREND_EMA_SLOW", 50)
        self._adx_period = config.get("TREND_ADX_PERIOD", 14)
        self._adx_threshold = config.get("TREND_ADX_THRESHOLD", 25)
        self._atr_mult_sl = config.get("TREND_ATR_MULT_SL", 1.5)
        self._tp_rr = config.get("TREND_TP_RR", 2.5)
        self._london_start = config.get("LONDON_START", 7)
        self._london_end = 10  # Only trade during 07:00-10:00 UTC opening hours

        # Pullback parameters (in pips)
        self._pullback_away_pips = 5.0   # Price was > 5 pips from EMA
        self._pullback_near_pips = 3.0   # Now within 3 pips of EMA

    @property
    def name(self) -> str:
        return "trend_following"

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    def _is_london_session(self, h1: pd.DataFrame) -> bool:
        """Check if the latest bar falls within the London open window (07:00-10:00 UTC)."""
        if len(h1) == 0:
            return False

        last_time = h1.index[-1]
        if hasattr(last_time, "hour"):
            hour = last_time.hour
            return self._london_start <= hour < self._london_end
        return False

    def _get_h4_trend(self, h4: pd.DataFrame) -> str:
        """Determine H4 trend direction from EMA crossover.

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if len(h4) < self._ema_slow + 1:
            return "neutral"

        ema_fast = h4.ta.ema(length=self._ema_fast)
        ema_slow = h4.ta.ema(length=self._ema_slow)

        if ema_fast is None or ema_slow is None:
            return "neutral"

        last_fast = ema_fast.iloc[-1]
        last_slow = ema_slow.iloc[-1]

        if np.isnan(last_fast) or np.isnan(last_slow):
            return "neutral"

        if last_fast > last_slow:
            return "bullish"
        elif last_fast < last_slow:
            return "bearish"
        return "neutral"

    def _check_h1_trend_alignment(self, h1: pd.DataFrame, h4_trend: str) -> bool:
        """Verify H1 EMAs are aligned with H4 trend and ADX > threshold."""
        if len(h1) < self._ema_slow + 1:
            return False

        ema_fast = h1.ta.ema(length=self._ema_fast)
        ema_slow = h1.ta.ema(length=self._ema_slow)
        adx_df = h1.ta.adx(length=self._adx_period)

        if ema_fast is None or ema_slow is None or adx_df is None:
            return False

        last_fast = ema_fast.iloc[-1]
        last_slow = ema_slow.iloc[-1]

        # ADX column name from pandas_ta: ADX_{period}
        adx_col = f"ADX_{self._adx_period}"
        if adx_col not in adx_df.columns:
            return False

        last_adx = adx_df[adx_col].iloc[-1]

        if np.isnan(last_fast) or np.isnan(last_slow) or np.isnan(last_adx):
            return False

        # ADX must indicate strong trend
        if last_adx < self._adx_threshold:
            return False

        # EMA alignment must match H4 trend
        if h4_trend == "bullish" and last_fast > last_slow:
            return True
        elif h4_trend == "bearish" and last_fast < last_slow:
            return True

        return False

    def _check_pullback(self, h1: pd.DataFrame, trend: str, symbol: str) -> bool:
        """Check for pullback to EMA20.

        Pullback condition:
          - At some recent point (within last 5 bars), price was > 5 pips from EMA20
          - Current bar's close is within 3 pips of EMA20
        """
        if len(h1) < self._ema_fast + 6:
            return False

        digits = _get_digits(symbol)
        ema = h1.ta.ema(length=self._ema_fast)
        if ema is None:
            return False

        current_close = h1["close"].iloc[-1]
        current_ema = ema.iloc[-1]

        if np.isnan(current_ema):
            return False

        away_price = _pips_to_price(self._pullback_away_pips, digits)
        near_price = _pips_to_price(self._pullback_near_pips, digits)

        # Check current bar is near EMA
        current_distance = abs(current_close - current_ema)
        if current_distance > near_price:
            return False

        # Check if price was far from EMA in recent bars (lookback 2-5 bars)
        for i in range(-5, -1):
            if abs(i) > len(h1):
                continue
            bar_close = h1["close"].iloc[i]
            bar_ema = ema.iloc[i]
            if np.isnan(bar_ema):
                continue

            distance = abs(bar_close - bar_ema)
            if distance > away_price:
                # Verify the direction was correct for the trend
                if trend == "bullish" and bar_close > bar_ema:
                    return True
                elif trend == "bearish" and bar_close < bar_ema:
                    return True

        return False

    def _compute_sl_tp_pips(self, h1: pd.DataFrame, symbol: str) -> tuple[float, float]:
        """Compute SL and TP in pips from H1 ATR.

        SL = ATR(14) * 1.5 converted to pips
        TP = SL * 2.5
        """
        digits = _get_digits(symbol)

        atr = h1.ta.atr(length=self._adx_period)
        if atr is None or len(atr) == 0:
            return 20.0, 50.0

        last_atr = atr.iloc[-1]
        if np.isnan(last_atr):
            return 20.0, 50.0

        sl_price_distance = last_atr * self._atr_mult_sl
        sl_pips = _points_to_pips(sl_price_distance, digits)

        # Clamp to reasonable range for H1 trades
        sl_pips = max(10.0, min(sl_pips, 80.0))
        tp_pips = round(sl_pips * self._tp_rr, 1)

        return round(sl_pips, 1), tp_pips

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(self, h1: pd.DataFrame, h4: pd.DataFrame,
                            trend: str, symbol: str) -> float:
        """Compute confidence score (0-100).

        Scoring:
          - H4 trend aligned:            +20
          - H1 EMA crossover aligned:    +20
          - ADX strength (above 25):     +20 (scaled, max at ADX 40+)
          - Pullback quality:            +20 (price near EMA + bouncing)
          - Multi-timeframe RSI align:   +20
        """
        score = 0.0

        # H4 trend present
        if trend in ("bullish", "bearish"):
            score += 20.0

        # H1 EMA aligned (already checked, so +20)
        score += 20.0

        # ADX strength bonus
        adx_df = h1.ta.adx(length=self._adx_period)
        if adx_df is not None:
            adx_col = f"ADX_{self._adx_period}"
            if adx_col in adx_df.columns:
                adx_val = adx_df[adx_col].iloc[-1]
                if not np.isnan(adx_val):
                    # Scale: ADX 25=0, ADX 40+=max 20
                    adx_bonus = min((adx_val - self._adx_threshold) / 15.0, 1.0) * 20.0
                    score += max(0.0, adx_bonus)

        # Pullback quality: how close to EMA
        digits = _get_digits(symbol)
        ema = h1.ta.ema(length=self._ema_fast)
        if ema is not None:
            current_close = h1["close"].iloc[-1]
            current_ema = ema.iloc[-1]
            if not np.isnan(current_ema):
                dist_pips = _points_to_pips(abs(current_close - current_ema), digits)
                # Closer to EMA = better (max score at 0 pips distance)
                pullback_score = max(0.0, 1.0 - dist_pips / self._pullback_near_pips) * 20.0
                score += pullback_score

        # H4 RSI alignment
        rsi_h4 = h4.ta.rsi(length=14) if len(h4) > 14 else None
        if rsi_h4 is not None and len(rsi_h4) > 0:
            rsi_val = rsi_h4.iloc[-1]
            if not np.isnan(rsi_val):
                if trend == "bullish" and rsi_val > 50:
                    score += min((rsi_val - 50) / 30, 1.0) * 20.0
                elif trend == "bearish" and rsi_val < 50:
                    score += min((50 - rsi_val) / 30, 1.0) * 20.0

        return round(min(score, 100.0), 1)

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(self, symbol: str, bars_dict: dict) -> dict | None:
        """Analyze market data and return a Signal dict or None.

        Args:
            symbol: e.g. "EURUSD"
            bars_dict: {"M5": DataFrame, "M15": DataFrame, "H1": DataFrame, "H4": DataFrame}

        Returns:
            Signal dict or None.
        """
        h1 = bars_dict.get("H1")
        h4 = bars_dict.get("H4")

        if h1 is None or h4 is None:
            logger.debug(f"[{self.name}] Missing H1 or H4 data for {symbol}")
            return None

        if len(h1) < self._ema_slow + 10 or len(h4) < self._ema_slow + 5:
            logger.debug(f"[{self.name}] Insufficient bars for {symbol}")
            return None

        # Step 1: Session filter - only trade during London open
        if not self._is_london_session(h1):
            logger.debug(f"[{self.name}] {symbol} outside London session window")
            return None

        # Step 2: Determine H4 trend
        h4_trend = self._get_h4_trend(h4)
        if h4_trend == "neutral":
            logger.debug(f"[{self.name}] {symbol} H4 trend neutral, skipping")
            return None

        # Step 3: H1 trend alignment + ADX filter
        if not self._check_h1_trend_alignment(h1, h4_trend):
            logger.debug(f"[{self.name}] {symbol} H1 not aligned with H4 trend or ADX too low")
            return None

        # Step 4: Pullback to EMA20
        if not self._check_pullback(h1, h4_trend, symbol):
            logger.debug(f"[{self.name}] {symbol} no pullback detected")
            return None

        # Step 5: Compute SL/TP
        sl_pips, tp_pips = self._compute_sl_tp_pips(h1, symbol)

        # Step 6: Confidence scoring
        confidence = self._compute_confidence(h1, h4, h4_trend, symbol)

        direction = "buy" if h4_trend == "bullish" else "sell"

        reasoning = (
            f"H4 trend={h4_trend} (EMA{self._ema_fast}/{self._ema_slow} crossover), "
            f"H1 trend aligned with ADX>{self._adx_threshold}, "
            f"pullback to EMA{self._ema_fast} detected, "
            f"London session active, "
            f"SL={sl_pips} pips (ATR*{self._atr_mult_sl}), TP={tp_pips} pips (1:{self._tp_rr} R:R)"
        )

        logger.info(f"[{self.name}] SIGNAL: {direction.upper()} {symbol} | "
                     f"confidence={confidence}")

        return {
            "symbol": symbol,
            "direction": direction,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "confidence": confidence,
            "strategy": self.name,
            "reasoning": reasoning,
        }
