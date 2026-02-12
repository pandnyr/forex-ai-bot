"""
Forex Session Agent - Market hours and session management
"""
import logging
from datetime import datetime, timezone
from ..config import (
    LONDON_START, LONDON_END, NY_START, NY_END,
    ASIAN_START, ASIAN_END, OVERLAP_START, OVERLAP_END,
    MT5_SYMBOLS
)

logger = logging.getLogger(__name__)


class ForexSessionAgent:
    def get_current_session(self) -> dict:
        """
        Determine the current trading session.
        Returns: {"session": str, "active": bool, "symbols": list, "mode": str}
        """
        now = datetime.now(timezone.utc)
        hour = now.hour
        day = now.weekday()  # 0=Monday, 6=Sunday

        # Weekend check (Friday 21:00 UTC to Sunday 22:00 UTC)
        if day == 5 or day == 6 or (day == 4 and hour >= 21):
            return {
                "session": "weekend",
                "active": False,
                "symbols": [],
                "mode": "off",
                "description": "Market closed - weekend",
            }

        # Determine session
        if OVERLAP_START <= hour < OVERLAP_END:
            return {
                "session": "london_ny_overlap",
                "active": True,
                "symbols": MT5_SYMBOLS,
                "mode": "aggressive",
                "description": "London-NY overlap - highest volatility",
            }
        elif LONDON_START <= hour < LONDON_END:
            return {
                "session": "london",
                "active": True,
                "symbols": ["EURUSD", "GBPUSD"],
                "mode": "aggressive",
                "description": "London session - EUR/GBP pairs preferred",
            }
        elif NY_START <= hour < NY_END:
            return {
                "session": "new_york",
                "active": True,
                "symbols": MT5_SYMBOLS,
                "mode": "normal",
                "description": "New York session",
            }
        elif ASIAN_START <= hour < ASIAN_END:
            return {
                "session": "asian",
                "active": True,
                "symbols": ["USDJPY"],
                "mode": "conservative",
                "description": "Asian session - USDJPY mean reversion only",
            }
        else:
            return {
                "session": "off_hours",
                "active": False,
                "symbols": [],
                "mode": "monitor",
                "description": "Off hours - monitor only, manage open positions",
            }

    def get_recommended_strategies(self, session: str, regime: str) -> list:
        """Get recommended strategies for current session and regime."""
        strategy_map = {
            "london_ny_overlap": {
                "trending": ["scalp_momentum", "trend_following"],
                "ranging": ["scalp_momentum", "mean_reversion"],
                "transitional": ["scalp_momentum", "breakout"],
            },
            "london": {
                "trending": ["trend_following", "scalp_momentum", "breakout"],
                "ranging": ["breakout", "scalp_momentum"],
                "transitional": ["breakout", "scalp_momentum"],
            },
            "new_york": {
                "trending": ["scalp_momentum", "trend_following"],
                "ranging": ["scalp_momentum"],
                "transitional": ["scalp_momentum"],
            },
            "asian": {
                "trending": [],
                "ranging": ["mean_reversion"],
                "transitional": ["mean_reversion"],
            },
        }

        session_strategies = strategy_map.get(session, {})
        return session_strategies.get(regime, [])

    def is_high_impact_news_window(self) -> bool:
        """
        Placeholder for news calendar integration.
        Returns True if within 30min of high-impact news.
        TODO: Integrate with forex factory calendar API
        """
        return False
