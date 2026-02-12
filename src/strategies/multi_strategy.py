"""
Multi-Strategy Orchestrator - Market Regime-Aware Strategy Selector
Combines all four strategies with adaptive weighting based on market regime.

Market Regime Detection (ADX on H1):
  - ADX > 25:         Trending     -> scalp 50%, trend 30%, breakout 20%
  - ADX < 20:         Ranging      -> mean_rev 60%, scalp 20%, breakout 20%
  - 20 <= ADX <= 25:  Transitional -> scalp 40%, breakout 30%, mean_rev 30%

Session-Aware:
  - Asian (00:00-06:00):   mean_reversion (primary), scalp_momentum
  - London (07:00-16:00):  trend_following, breakout, scalp_momentum
  - New York (13:00-21:00): scalp_momentum, trend_following
"""
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pandas_ta as ta

from .scalp_momentum import ScalpMomentumStrategy
from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy

logger = logging.getLogger(__name__)

# Regime weight allocation: {regime: {strategy_name: weight}}
REGIME_WEIGHTS = {
    "trending": {
        "scalp_momentum": 0.50,
        "trend_following": 0.30,
        "breakout": 0.20,
        "mean_reversion": 0.00,
    },
    "ranging": {
        "mean_reversion": 0.60,
        "scalp_momentum": 0.20,
        "breakout": 0.20,
        "trend_following": 0.00,
    },
    "transitional": {
        "scalp_momentum": 0.40,
        "breakout": 0.30,
        "mean_reversion": 0.30,
        "trend_following": 0.00,
    },
}

# Session-strategy eligibility
SESSION_STRATEGIES = {
    "asian": ["mean_reversion", "scalp_momentum"],
    "london": ["trend_following", "breakout", "scalp_momentum"],
    "new_york": ["scalp_momentum", "trend_following"],
    "overlap": ["scalp_momentum", "trend_following", "breakout"],
}


class MultiStrategyOrchestrator:
    """Orchestrates multiple strategies with regime and session awareness."""

    def __init__(self, config: dict):
        """
        Initialize all sub-strategies and orchestrator parameters.

        Expected config keys (in addition to each strategy's keys):
            SCALP_MIN_AI_CONFIDENCE (float): Minimum confidence threshold, default 65
            TREND_ADX_PERIOD (int): ADX period for regime detection, default 14
            ASIAN_START (int): Asian session start, default 0
            ASIAN_END (int): Asian session end, default 8
            LONDON_START (int): London session start, default 7
            LONDON_END (int): London session end, default 16
            NY_START (int): New York session start, default 13
            NY_END (int): New York session end, default 21
            OVERLAP_START (int): Overlap session start, default 13
            OVERLAP_END (int): Overlap session end, default 16
        """
        self._config = config
        self._min_confidence = config.get("SCALP_MIN_AI_CONFIDENCE", 65)
        self._adx_period = config.get("TREND_ADX_PERIOD", 14)

        # Session boundaries (UTC hours)
        self._asian_start = config.get("ASIAN_START", 0)
        self._asian_end = config.get("ASIAN_END", 8)
        self._london_start = config.get("LONDON_START", 7)
        self._london_end = config.get("LONDON_END", 16)
        self._ny_start = config.get("NY_START", 13)
        self._ny_end = config.get("NY_END", 21)
        self._overlap_start = config.get("OVERLAP_START", 13)
        self._overlap_end = config.get("OVERLAP_END", 16)

        # Initialize sub-strategies
        self._strategies = {
            "scalp_momentum": ScalpMomentumStrategy(config),
            "trend_following": TrendFollowingStrategy(config),
            "mean_reversion": MeanReversionStrategy(config),
            "breakout": BreakoutStrategy(config),
        }

        logger.info(f"[multi_strategy] Initialized with {len(self._strategies)} strategies, "
                     f"min_confidence={self._min_confidence}")

    @property
    def name(self) -> str:
        return "multi_strategy"

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def detect_regime(self, h1_bars: pd.DataFrame) -> str:
        """Detect market regime using ADX on H1.

        Returns:
            'trending'     if ADX > 25
            'ranging'      if ADX < 20
            'transitional' if 20 <= ADX <= 25
        """
        if h1_bars is None or len(h1_bars) < self._adx_period + 5:
            logger.debug("[multi_strategy] Insufficient H1 data for regime detection, "
                         "defaulting to 'transitional'")
            return "transitional"

        adx_df = h1_bars.ta.adx(length=self._adx_period)
        if adx_df is None:
            return "transitional"

        adx_col = f"ADX_{self._adx_period}"
        if adx_col not in adx_df.columns:
            return "transitional"

        last_adx = adx_df[adx_col].iloc[-1]
        if np.isnan(last_adx):
            return "transitional"

        if last_adx > 25:
            regime = "trending"
        elif last_adx < 20:
            regime = "ranging"
        else:
            regime = "transitional"

        logger.debug(f"[multi_strategy] Regime: {regime} (ADX={last_adx:.1f})")
        return regime

    # ------------------------------------------------------------------
    # Session detection
    # ------------------------------------------------------------------

    def _detect_session(self, hour: int = None) -> str:
        """Detect current trading session based on UTC hour.

        Args:
            hour: UTC hour (0-23). If None, uses current time.

        Returns:
            'asian', 'london', 'new_york', or 'overlap'
        """
        if hour is None:
            hour = datetime.now(timezone.utc).hour

        # Overlap takes priority (London + New York)
        if self._overlap_start <= hour < self._overlap_end:
            return "overlap"
        elif self._london_start <= hour < self._london_end:
            return "london"
        elif self._ny_start <= hour < self._ny_end:
            return "new_york"
        elif self._asian_start <= hour < self._asian_end:
            return "asian"
        else:
            # Late NY / early Asian gap
            return "asian"

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def get_active_strategies(self, session: str, regime: str) -> list[str]:
        """Determine which strategies should be active for the current session and regime.

        A strategy is active if:
          1. It is eligible for the current session
          2. Its regime weight is > 0

        Returns:
            List of strategy names to run, ordered by regime weight (highest first).
        """
        session_eligible = set(SESSION_STRATEGIES.get(session, []))
        regime_weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["transitional"])

        active = []
        for strat_name, weight in regime_weights.items():
            if weight > 0 and strat_name in session_eligible:
                active.append((strat_name, weight))

        # Sort by weight descending
        active.sort(key=lambda x: x[1], reverse=True)

        strategy_names = [name for name, _ in active]
        logger.debug(f"[multi_strategy] Active strategies for session={session}, "
                     f"regime={regime}: {strategy_names}")
        return strategy_names

    # ------------------------------------------------------------------
    # Signal weighting
    # ------------------------------------------------------------------

    def _apply_regime_weight(self, signal: dict, regime: str) -> dict:
        """Apply regime-based weight to signal confidence.

        The raw confidence from the strategy is multiplied by the regime weight
        to produce a weighted confidence score.
        """
        weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["transitional"])
        strategy_name = signal.get("strategy", "")
        weight = weights.get(strategy_name, 0.0)

        if weight == 0:
            return signal

        # Weighted confidence: base confidence * weight (but keep within 0-100)
        raw_confidence = signal.get("confidence", 0.0)
        weighted_confidence = round(raw_confidence * (0.5 + weight * 0.5), 1)
        # The formula: at weight=1.0, full confidence; at weight=0.2, 60% of confidence

        signal_copy = signal.copy()
        signal_copy["confidence"] = min(100.0, weighted_confidence)
        signal_copy["reasoning"] = (
            f"[regime={regime}, weight={weight:.0%}] " + signal.get("reasoning", "")
        )
        return signal_copy

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def get_signals(self, symbol: str, bars_dict: dict,
                    session: str = None) -> list[dict]:
        """Run relevant strategies and return sorted signals.

        Args:
            symbol: e.g. "EURUSD"
            bars_dict: {"M5": DataFrame, "M15": DataFrame, "H1": DataFrame, "H4": DataFrame}
            session: Override session detection ('asian', 'london', 'new_york', 'overlap').
                     If None, auto-detected from current UTC time.

        Returns:
            List of Signal dicts, sorted by confidence (highest first),
            filtered by MIN_AI_CONFIDENCE threshold.
        """
        h1 = bars_dict.get("H1")

        # Detect regime
        regime = self.detect_regime(h1)

        # Detect or use provided session
        if session is None:
            session = self._detect_session()

        # Get active strategies
        active_names = self.get_active_strategies(session, regime)

        if not active_names:
            logger.debug(f"[multi_strategy] No active strategies for {symbol} "
                         f"(session={session}, regime={regime})")
            return []

        # Run each active strategy
        signals = []
        for strat_name in active_names:
            strategy = self._strategies.get(strat_name)
            if strategy is None:
                continue

            try:
                signal = strategy.analyze(symbol, bars_dict)
                if signal is not None:
                    # Apply regime weight to confidence
                    weighted_signal = self._apply_regime_weight(signal, regime)
                    signals.append(weighted_signal)
                    logger.debug(f"[multi_strategy] {strat_name} produced signal for {symbol}: "
                                 f"confidence={weighted_signal['confidence']}")
            except Exception as e:
                logger.error(f"[multi_strategy] Error running {strat_name} for {symbol}: {e}",
                             exc_info=True)

        # Filter by minimum confidence
        signals = [s for s in signals if s["confidence"] >= self._min_confidence]

        # Sort by confidence (highest first)
        signals.sort(key=lambda s: s["confidence"], reverse=True)

        if signals:
            logger.info(f"[multi_strategy] {symbol}: {len(signals)} signal(s) "
                        f"(session={session}, regime={regime}), "
                        f"best={signals[0]['strategy']} confidence={signals[0]['confidence']}")
        else:
            logger.debug(f"[multi_strategy] {symbol}: no signals above threshold "
                         f"(session={session}, regime={regime})")

        return signals

    def analyze(self, symbol: str, bars_dict: dict) -> dict | None:
        """Analyze and return the best signal (highest confidence).

        This method provides the same interface as individual strategies,
        making MultiStrategyOrchestrator interchangeable.

        Args:
            symbol: e.g. "EURUSD"
            bars_dict: {"M5": DataFrame, "M15": DataFrame, "H1": DataFrame, "H4": DataFrame}

        Returns:
            Best Signal dict or None.
        """
        signals = self.get_signals(symbol, bars_dict)
        if not signals:
            return None
        return signals[0]
