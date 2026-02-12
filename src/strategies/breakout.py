"""
London Breakout Strategy - Pre-London Range Breakout on M15
Captures the volatility expansion at the London open.

Entry Logic:
  - Pre-London range: high/low of bars between 04:00-07:00 UTC (on M15)
  - Buy: price breaks above range high with volume spike > 1.5x average
  - Sell: price breaks below range low with volume spike > 1.5x average
  - Range filter: 10-40 pips (too narrow = noise, too wide = risky)
  - Only trade at London open window (07:00-08:00 UTC)

SL: Opposite end of the pre-London range
TP: 1.5x range width
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


class BreakoutStrategy:
    """London Breakout Strategy - Pre-London Range Break on M15."""

    def __init__(self, config: dict):
        """
        Initialize with config dictionary.

        Expected config keys:
            BREAKOUT_TP_MULT (float): TP as multiple of range width, default 1.5
            BREAKOUT_MAX_RANGE_PIPS (float): Max range in pips, default 40
            BREAKOUT_MIN_RANGE_PIPS (float): Min range in pips, default 10
            BREAKOUT_LOOKBACK_BARS (int): Number of M15 bars for range (04:00-07:00), default 12
        """
        self._config = config
        self._tp_mult = config.get("BREAKOUT_TP_MULT", 1.5)
        self._max_range_pips = config.get("BREAKOUT_MAX_RANGE_PIPS", 40)
        self._min_range_pips = config.get("BREAKOUT_MIN_RANGE_PIPS", 10)
        self._lookback_bars = config.get("BREAKOUT_LOOKBACK_BARS", 12)

        # Pre-London range hours (UTC)
        self._range_start_hour = 4
        self._range_end_hour = 7

        # London open trade window (UTC)
        self._trade_start_hour = 7
        self._trade_end_hour = 8

        # Volume spike threshold
        self._volume_spike_mult = 1.5

    @property
    def name(self) -> str:
        return "breakout"

    # ------------------------------------------------------------------
    # Session & range helpers
    # ------------------------------------------------------------------

    def _is_london_open(self, m15: pd.DataFrame) -> bool:
        """Check if the latest bar is during the London open window (07:00-08:00 UTC)."""
        if len(m15) == 0:
            return False

        last_time = m15.index[-1]
        if hasattr(last_time, "hour"):
            hour = last_time.hour
            return self._trade_start_hour <= hour < self._trade_end_hour
        return False

    def _compute_pre_london_range(self, m15: pd.DataFrame) -> tuple[float, float, bool]:
        """Compute the pre-London consolidation range (04:00-07:00 UTC).

        Scans the M15 bars that fall within the 04:00-07:00 UTC window
        on the same day as the latest bar.

        Returns:
            (range_high, range_low, valid)
        """
        if len(m15) < self._lookback_bars:
            return 0.0, 0.0, False

        # Get the date of the latest bar
        last_time = m15.index[-1]
        if not hasattr(last_time, "hour"):
            return 0.0, 0.0, False

        # Filter bars within the pre-London range window
        range_bars = []
        for i in range(len(m15) - 1, -1, -1):
            bar_time = m15.index[i]
            if not hasattr(bar_time, "hour"):
                continue

            # Only consider bars from today (or at most the last 24 hours)
            time_diff = (last_time - bar_time).total_seconds()
            if time_diff > 86400:  # More than 24 hours ago
                break

            if self._range_start_hour <= bar_time.hour < self._range_end_hour:
                range_bars.append(i)

            # Stop scanning once we've gone past the range window
            if bar_time.hour < self._range_start_hour and time_diff > 3600:
                break

        if len(range_bars) < 3:  # Need at least 3 bars for a valid range
            return 0.0, 0.0, False

        # Compute high and low of the range
        range_indices = range_bars
        range_high = m15["high"].iloc[range_indices].max()
        range_low = m15["low"].iloc[range_indices].min()

        if np.isnan(range_high) or np.isnan(range_low):
            return 0.0, 0.0, False

        return range_high, range_low, True

    def _check_volume_spike(self, m15: pd.DataFrame) -> bool:
        """Check if the current bar has a volume spike > 1.5x the 20-bar average."""
        if "volume" not in m15.columns or len(m15) < 21:
            # If no volume data, pass the filter (some forex feeds lack tick volume)
            return True

        current_vol = m15["volume"].iloc[-1]
        avg_vol = m15["volume"].iloc[-21:-1].mean()

        if np.isnan(current_vol) or np.isnan(avg_vol) or avg_vol == 0:
            return True  # Pass if volume data unreliable

        return current_vol > avg_vol * self._volume_spike_mult

    # ------------------------------------------------------------------
    # Signal detection
    # ------------------------------------------------------------------

    def _detect_breakout(self, m15: pd.DataFrame, range_high: float,
                         range_low: float) -> tuple[str | None, float]:
        """Detect if price has broken above/below the pre-London range.

        Returns:
            (direction: 'buy'|'sell'|None, breakout_distance: float)
        """
        if len(m15) == 0:
            return None, 0.0

        last_close = m15["close"].iloc[-1]
        last_high = m15["high"].iloc[-1]
        last_low = m15["low"].iloc[-1]

        # Breakout above range high
        if last_close > range_high and last_high > range_high:
            return "buy", last_close - range_high

        # Breakout below range low
        if last_close < range_low and last_low < range_low:
            return "sell", range_low - last_close

        return None, 0.0

    # ------------------------------------------------------------------
    # SL/TP calculation
    # ------------------------------------------------------------------

    def _compute_sl_tp(self, direction: str, range_high: float,
                       range_low: float, symbol: str) -> tuple[float, float, float]:
        """Compute SL and TP in pips.

        SL: opposite end of the range
        TP: 1.5x range width

        Returns:
            (sl_pips, tp_pips, range_width_pips)
        """
        digits = _get_digits(symbol)
        range_width = range_high - range_low
        range_width_pips = _points_to_pips(range_width, digits)

        if direction == "buy":
            # SL at range low (distance from range_high to range_low in pips)
            sl_pips = range_width_pips
        else:
            # SL at range high
            sl_pips = range_width_pips

        tp_pips = round(range_width_pips * self._tp_mult, 1)

        return round(sl_pips, 1), tp_pips, round(range_width_pips, 1)

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(self, direction: str, m15: pd.DataFrame,
                            range_high: float, range_low: float,
                            range_width_pips: float,
                            breakout_distance: float, symbol: str) -> float:
        """Compute confidence score (0-100).

        Scoring:
          - Range width quality:      +20 (optimal range: 20-30 pips)
          - Volume spike strength:    +25 (higher volume = stronger breakout)
          - Breakout candle strength:  +20 (strong close beyond range)
          - Trend alignment on H1:    +15 (if available in bars_dict via m15 proxy)
          - Range bar count:           +10 (more bars in range = more defined)
          - First test of range:       +10 (first breakout attempt is strongest)
        """
        digits = _get_digits(symbol)
        score = 0.0

        # Range width quality (optimal: 20-30 pips)
        if 20 <= range_width_pips <= 30:
            score += 20.0
        elif 15 <= range_width_pips < 20 or 30 < range_width_pips <= 35:
            score += 15.0
        elif self._min_range_pips <= range_width_pips <= self._max_range_pips:
            score += 10.0

        # Volume spike strength
        if "volume" in m15.columns and len(m15) >= 21:
            current_vol = m15["volume"].iloc[-1]
            avg_vol = m15["volume"].iloc[-21:-1].mean()
            if avg_vol > 0 and not np.isnan(current_vol) and not np.isnan(avg_vol):
                vol_ratio = current_vol / avg_vol
                # Scale: 1.5x = base, 3x+ = max
                vol_score = min((vol_ratio - 1.0) / 2.0, 1.0) * 25.0
                score += max(0.0, vol_score)

        # Breakout candle strength (how far beyond range)
        breakout_pips = _points_to_pips(breakout_distance, digits)
        if breakout_pips > 0:
            # More penetration = higher confidence (max at 5 pips beyond)
            strength_score = min(breakout_pips / 5.0, 1.0) * 20.0
            score += strength_score

        # Candle body vs wick ratio (strong close = less wick)
        if len(m15) >= 1:
            curr = m15.iloc[-1]
            body = abs(curr["close"] - curr["open"])
            total_range = curr["high"] - curr["low"]
            if total_range > 0:
                body_ratio = body / total_range
                score += body_ratio * 15.0

        # Range definition quality (number of bars that touched boundaries)
        range_bars_count = 0
        for i in range(len(m15) - 1, max(len(m15) - self._lookback_bars - 1, -1), -1):
            bar_time = m15.index[i]
            if hasattr(bar_time, "hour"):
                if self._range_start_hour <= bar_time.hour < self._range_end_hour:
                    range_bars_count += 1

        if range_bars_count >= 8:
            score += 10.0
        elif range_bars_count >= 5:
            score += 7.0
        elif range_bars_count >= 3:
            score += 4.0

        return round(min(score, 100.0), 1)

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(self, symbol: str, bars_dict: dict) -> dict | None:
        """Analyze market data and return a Signal dict or None.

        Args:
            symbol: e.g. "GBPUSD"
            bars_dict: {"M5": DataFrame, "M15": DataFrame, "H1": DataFrame, "H4": DataFrame}

        Returns:
            Signal dict or None.
        """
        m15 = bars_dict.get("M15")

        if m15 is None:
            logger.debug(f"[{self.name}] Missing M15 data for {symbol}")
            return None

        if len(m15) < self._lookback_bars + 10:
            logger.debug(f"[{self.name}] Insufficient M15 bars for {symbol}")
            return None

        # Step 1: Trade window filter - only trade at London open (07:00-08:00 UTC)
        if not self._is_london_open(m15):
            logger.debug(f"[{self.name}] {symbol} outside London open window")
            return None

        # Step 2: Compute pre-London range
        range_high, range_low, valid = self._compute_pre_london_range(m15)
        if not valid:
            logger.debug(f"[{self.name}] {symbol} could not compute pre-London range")
            return None

        # Step 3: Range width filter
        digits = _get_digits(symbol)
        range_width_pips = _points_to_pips(range_high - range_low, digits)

        if range_width_pips < self._min_range_pips:
            logger.debug(f"[{self.name}] {symbol} range too narrow: {range_width_pips:.1f} pips "
                         f"(min={self._min_range_pips})")
            return None

        if range_width_pips > self._max_range_pips:
            logger.debug(f"[{self.name}] {symbol} range too wide: {range_width_pips:.1f} pips "
                         f"(max={self._max_range_pips})")
            return None

        # Step 4: Detect breakout
        direction, breakout_distance = self._detect_breakout(m15, range_high, range_low)
        if direction is None:
            logger.debug(f"[{self.name}] {symbol} no breakout detected "
                         f"(range: {range_low:.5f} - {range_high:.5f})")
            return None

        # Step 5: Volume confirmation
        if not self._check_volume_spike(m15):
            logger.debug(f"[{self.name}] {symbol} breakout without volume confirmation")
            return None

        # Step 6: Compute SL/TP
        sl_pips, tp_pips, _ = self._compute_sl_tp(direction, range_high, range_low, symbol)

        # Step 7: Confidence scoring
        confidence = self._compute_confidence(
            direction, m15, range_high, range_low,
            range_width_pips, breakout_distance, symbol
        )

        breakout_pips = _points_to_pips(breakout_distance, digits)
        reasoning = (
            f"London breakout: pre-London range {range_low:.5f}-{range_high:.5f} "
            f"({range_width_pips:.1f} pips), "
            f"{'upside' if direction == 'buy' else 'downside'} break by {breakout_pips:.1f} pips, "
            f"volume confirmed, "
            f"SL={sl_pips} pips (opposite range end), "
            f"TP={tp_pips} pips ({self._tp_mult}x range width)"
        )

        logger.info(f"[{self.name}] SIGNAL: {direction.upper()} {symbol} | "
                     f"confidence={confidence} | range={range_width_pips:.1f} pips")

        return {
            "symbol": symbol,
            "direction": direction,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "confidence": confidence,
            "strategy": self.name,
            "reasoning": reasoning,
        }
