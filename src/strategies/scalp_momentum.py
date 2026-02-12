"""
Scalp Momentum Strategy - M5/M15 Momentum Scalping
Primary strategy for the Forex AI Trading Bot.

Entry Logic:
  - H1 bias: EMA20 > EMA50 = bullish (only longs), opposite for bearish
  - M15 momentum: RSI > 50 AND price > EMA20 (for longs)
  - M5 trigger: bullish engulfing OR pin bar at EMA20 level
  - Spread filter: only trade when spread < MAX_SPREAD_PIPS

SL: ATR(14) * 1.5 on M5 (converted to pips)
TP: SL * 2.0 (1:2 R:R)
"""
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def _points_to_pips(points: float, digits: int = 5) -> float:
    """Convert price difference to pips.
    For 5/3-digit brokers: 1 pip = 10 points (0.00010 for 5-digit, 0.010 for 3-digit).
    For 4/2-digit brokers: 1 pip = 1 point.
    """
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
        return 3  # e.g., USDJPY = 3 digits (5-digit broker: 142.123)
    return 5  # Most major pairs: 5 digits (1.12345)


class ScalpMomentumStrategy:
    """M5/M15 Momentum Scalping Strategy."""

    def __init__(self, config: dict):
        """
        Initialize with config dictionary.

        Expected config keys:
            SCALP_EMA_FAST (int): Fast EMA period, default 20
            SCALP_EMA_SLOW (int): Slow EMA period, default 50
            SCALP_RSI_PERIOD (int): RSI period, default 14
            SCALP_RSI_THRESHOLD (float): RSI threshold for momentum, default 50
            SCALP_ATR_PERIOD (int): ATR period, default 14
            SCALP_SL_ATR_MULT (float): SL multiplier for ATR, default 1.5
            SCALP_TP_RR (float): Risk:Reward ratio for TP, default 2.0
            SCALP_MAX_SPREAD_PIPS (float): Max spread in pips, default 1.5
            SCALP_MIN_AI_CONFIDENCE (float): Min confidence score, default 65
        """
        self._config = config
        self._ema_fast = config.get("SCALP_EMA_FAST", 20)
        self._ema_slow = config.get("SCALP_EMA_SLOW", 50)
        self._rsi_period = config.get("SCALP_RSI_PERIOD", 14)
        self._rsi_threshold = config.get("SCALP_RSI_THRESHOLD", 50)
        self._atr_period = config.get("SCALP_ATR_PERIOD", 14)
        self._sl_atr_mult = config.get("SCALP_SL_ATR_MULT", 1.5)
        self._tp_rr = config.get("SCALP_TP_RR", 2.0)
        self._max_spread_pips = config.get("SCALP_MAX_SPREAD_PIPS", 1.5)
        self._min_confidence = config.get("SCALP_MIN_AI_CONFIDENCE", 65)

    @property
    def name(self) -> str:
        return "scalp_momentum"

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    def _compute_h1_bias(self, h1: pd.DataFrame) -> str:
        """Determine H1 directional bias from EMA crossover.

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if len(h1) < self._ema_slow + 1:
            return "neutral"

        ema_fast = h1.ta.ema(length=self._ema_fast)
        ema_slow = h1.ta.ema(length=self._ema_slow)

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

    def _check_m15_momentum(self, m15: pd.DataFrame, bias: str) -> bool:
        """Check M15 momentum confirmation.

        For longs:  RSI > threshold AND price > EMA20
        For shorts: RSI < (100 - threshold) AND price < EMA20
        """
        if len(m15) < max(self._ema_fast, self._rsi_period) + 1:
            return False

        rsi = m15.ta.rsi(length=self._rsi_period)
        ema = m15.ta.ema(length=self._ema_fast)

        if rsi is None or ema is None:
            return False

        last_rsi = rsi.iloc[-1]
        last_ema = ema.iloc[-1]
        last_close = m15["close"].iloc[-1]

        if np.isnan(last_rsi) or np.isnan(last_ema):
            return False

        if bias == "bullish":
            return last_rsi > self._rsi_threshold and last_close > last_ema
        elif bias == "bearish":
            return last_rsi < (100 - self._rsi_threshold) and last_close < last_ema
        return False

    def _is_bullish_engulfing(self, m5: pd.DataFrame) -> bool:
        """Detect bullish engulfing pattern on last two M5 candles.

        Bullish engulfing:
          - Previous candle is bearish (open > close)
          - Current candle opens below previous close AND closes above previous open
        """
        if len(m5) < 2:
            return False

        prev = m5.iloc[-2]
        curr = m5.iloc[-1]

        prev_bearish = prev["open"] > prev["close"]
        engulfing = curr["open"] < prev["close"] and curr["close"] > prev["open"]

        return prev_bearish and engulfing

    def _is_bearish_engulfing(self, m5: pd.DataFrame) -> bool:
        """Detect bearish engulfing pattern on last two M5 candles.

        Bearish engulfing:
          - Previous candle is bullish (close > open)
          - Current candle opens above previous close AND closes below previous open
        """
        if len(m5) < 2:
            return False

        prev = m5.iloc[-2]
        curr = m5.iloc[-1]

        prev_bullish = prev["close"] > prev["open"]
        engulfing = curr["open"] > prev["close"] and curr["close"] < prev["open"]

        return prev_bullish and engulfing

    def _is_bullish_pin_bar(self, m5: pd.DataFrame, ema_value: float) -> bool:
        """Detect bullish pin bar near EMA20 on M5.

        Bullish pin bar:
          - Lower wick > 2x body size
          - Small body (upper wick is small relative to lower wick)
          - Price is near EMA20 level (within 5 pips)
        """
        if len(m5) < 1 or np.isnan(ema_value):
            return False

        curr = m5.iloc[-1]
        body = abs(curr["close"] - curr["open"])
        lower_wick = min(curr["open"], curr["close"]) - curr["low"]
        upper_wick = curr["high"] - max(curr["open"], curr["close"])

        if body == 0:
            body = 1e-10  # Avoid division by zero (doji is acceptable)

        pin_bar = lower_wick > 2.0 * body and lower_wick > upper_wick
        near_ema = abs(curr["low"] - ema_value) / ema_value < 0.003  # Within ~0.3%

        return pin_bar and near_ema

    def _is_bearish_pin_bar(self, m5: pd.DataFrame, ema_value: float) -> bool:
        """Detect bearish pin bar near EMA20 on M5.

        Bearish pin bar:
          - Upper wick > 2x body size
          - Price is near EMA20 level
        """
        if len(m5) < 1 or np.isnan(ema_value):
            return False

        curr = m5.iloc[-1]
        body = abs(curr["close"] - curr["open"])
        upper_wick = curr["high"] - max(curr["open"], curr["close"])
        lower_wick = min(curr["open"], curr["close"]) - curr["low"]

        if body == 0:
            body = 1e-10

        pin_bar = upper_wick > 2.0 * body and upper_wick > lower_wick
        near_ema = abs(curr["high"] - ema_value) / ema_value < 0.003

        return pin_bar and near_ema

    def _check_m5_trigger(self, m5: pd.DataFrame, bias: str) -> tuple[bool, str]:
        """Check for M5 trigger patterns.

        Returns:
            (triggered: bool, pattern: str)
        """
        if len(m5) < max(self._ema_fast, 2) + 1:
            return False, ""

        ema = m5.ta.ema(length=self._ema_fast)
        if ema is None:
            return False, ""

        last_ema = ema.iloc[-1]

        if bias == "bullish":
            if self._is_bullish_engulfing(m5):
                return True, "bullish_engulfing"
            if self._is_bullish_pin_bar(m5, last_ema):
                return True, "bullish_pin_bar"
        elif bias == "bearish":
            if self._is_bearish_engulfing(m5):
                return True, "bearish_engulfing"
            if self._is_bearish_pin_bar(m5, last_ema):
                return True, "bearish_pin_bar"

        return False, ""

    def _compute_sl_tp_pips(self, m5: pd.DataFrame, symbol: str) -> tuple[float, float]:
        """Compute SL and TP in pips from M5 ATR.

        SL = ATR(14) * 1.5 converted to pips
        TP = SL * TP_RR
        """
        digits = _get_digits(symbol)

        atr = m5.ta.atr(length=self._atr_period)
        if atr is None or len(atr) == 0:
            return 15.0, 30.0  # Fallback defaults

        last_atr = atr.iloc[-1]
        if np.isnan(last_atr):
            return 15.0, 30.0

        sl_price_distance = last_atr * self._sl_atr_mult
        sl_pips = _points_to_pips(sl_price_distance, digits)

        # Clamp SL to reasonable range
        sl_pips = max(5.0, min(sl_pips, 50.0))
        tp_pips = round(sl_pips * self._tp_rr, 1)

        return round(sl_pips, 1), tp_pips

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(self, bias: str, momentum_ok: bool,
                            pattern: str, m5: pd.DataFrame,
                            m15: pd.DataFrame) -> float:
        """Compute a confidence score (0-100) for the signal.

        Scoring:
          - H1 bias alignment:       +25
          - M15 momentum confirmed:  +25
          - M5 trigger pattern:      +20 (engulfing) or +15 (pin bar)
          - RSI strength on M15:     +15 (scaled by how far from 50)
          - ATR stability:           +15 (lower recent ATR volatility = more stable)
        """
        score = 0.0

        # H1 bias present
        if bias in ("bullish", "bearish"):
            score += 25.0

        # M15 momentum confirmed
        if momentum_ok:
            score += 25.0

        # M5 trigger pattern quality
        if "engulfing" in pattern:
            score += 20.0
        elif "pin_bar" in pattern:
            score += 15.0

        # RSI strength on M15
        rsi = m15.ta.rsi(length=self._rsi_period)
        if rsi is not None and len(rsi) > 0 and not np.isnan(rsi.iloc[-1]):
            rsi_val = rsi.iloc[-1]
            if bias == "bullish":
                # The further above 50, the stronger
                rsi_strength = min((rsi_val - 50) / 30, 1.0) * 15.0
            else:
                rsi_strength = min((50 - rsi_val) / 30, 1.0) * 15.0
            score += max(0.0, rsi_strength)

        # ATR stability: compare last ATR to rolling mean
        atr = m5.ta.atr(length=self._atr_period)
        if atr is not None and len(atr) >= 20:
            recent_atr = atr.iloc[-1]
            avg_atr = atr.iloc[-20:].mean()
            if avg_atr > 0 and not np.isnan(recent_atr) and not np.isnan(avg_atr):
                ratio = recent_atr / avg_atr
                # Stable ATR (ratio close to 1.0) gets more points
                stability = max(0.0, 1.0 - abs(ratio - 1.0)) * 15.0
                score += stability

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
            Signal dict with keys:
                symbol, direction, sl_pips, tp_pips, confidence,
                strategy, reasoning
            or None if no valid signal.
        """
        m5 = bars_dict.get("M5")
        m15 = bars_dict.get("M15")
        h1 = bars_dict.get("H1")

        # Validate data availability
        if m5 is None or m15 is None or h1 is None:
            logger.debug(f"[{self.name}] Missing timeframe data for {symbol}")
            return None

        if len(m5) < self._ema_slow + 5 or len(m15) < self._ema_slow + 5 or len(h1) < self._ema_slow + 5:
            logger.debug(f"[{self.name}] Insufficient bars for {symbol}")
            return None

        # Step 1: Determine H1 bias
        bias = self._compute_h1_bias(h1)
        if bias == "neutral":
            logger.debug(f"[{self.name}] {symbol} H1 bias neutral, skipping")
            return None

        # Step 2: Check M15 momentum
        momentum_ok = self._check_m15_momentum(m15, bias)
        if not momentum_ok:
            logger.debug(f"[{self.name}] {symbol} M15 momentum not confirmed for {bias}")
            return None

        # Step 3: Check M5 trigger
        triggered, pattern = self._check_m5_trigger(m5, bias)
        if not triggered:
            logger.debug(f"[{self.name}] {symbol} no M5 trigger for {bias}")
            return None

        # Step 4: Compute SL/TP
        sl_pips, tp_pips = self._compute_sl_tp_pips(m5, symbol)

        # Step 5: Confidence scoring
        confidence = self._compute_confidence(bias, momentum_ok, pattern, m5, m15)
        if confidence < self._min_confidence:
            logger.debug(f"[{self.name}] {symbol} confidence {confidence} below threshold {self._min_confidence}")
            return None

        direction = "buy" if bias == "bullish" else "sell"

        reasoning = (
            f"H1 bias={bias} (EMA{self._ema_fast}>{self._ema_slow}), "
            f"M15 momentum confirmed (RSI & price above EMA{self._ema_fast}), "
            f"M5 trigger={pattern}, "
            f"SL={sl_pips} pips (ATR*{self._sl_atr_mult}), TP={tp_pips} pips (1:{self._tp_rr} R:R)"
        )

        logger.info(f"[{self.name}] SIGNAL: {direction.upper()} {symbol} | "
                     f"confidence={confidence} | {pattern}")

        return {
            "symbol": symbol,
            "direction": direction,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "confidence": confidence,
            "strategy": self.name,
            "reasoning": reasoning,
        }
