"""
Risk Agent - 5-Stage Circuit Breaker
Adapted from MoonDev's risk management pattern for Forex
"""
import logging
from datetime import datetime, timezone
from ..config import (
    MIN_ACCOUNT_BALANCE, MAX_DAILY_LOSS_PCT, MAX_DAILY_PROFIT_PCT,
    RISK_PER_TRADE, MAX_CONCURRENT_POSITIONS, MAX_DAILY_TRADES,
    MAX_TOTAL_DD_PCT, MT5_MAGIC_NUMBER, CONSECUTIVE_LOSS_REDUCE,
    CONSECUTIVE_WIN_INCREASE,
)
from .. import nice_funcs_mt5 as mt5f

logger = logging.getLogger(__name__)


class RiskAgent:
    def __init__(self):
        self.peak_balance = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self._lot_multiplier = 1.0

    def check_all(self) -> dict:
        """
        Run all 5 circuit breaker checks.
        Returns: {"allowed": bool, "reason": str, "lot_multiplier": float}
        """
        balance = mt5f.get_account_balance()
        equity = mt5f.get_account_equity()

        # Track peak balance for drawdown calc
        if balance > self.peak_balance:
            self.peak_balance = balance

        # Stage 1: Minimum balance
        if balance < MIN_ACCOUNT_BALANCE:
            return self._block(f"Balance ${balance:.2f} below minimum ${MIN_ACCOUNT_BALANCE}")

        # Stage 2: Daily loss limit
        daily_pnl = mt5f.get_daily_pnl(MT5_MAGIC_NUMBER)
        daily_loss_pct = abs(min(0, daily_pnl)) / balance * 100
        if daily_loss_pct >= MAX_DAILY_LOSS_PCT:
            return self._block(f"Daily loss {daily_loss_pct:.1f}% >= limit {MAX_DAILY_LOSS_PCT}%")

        # Stage 3: Daily profit limit (overtrading prevention)
        daily_profit_pct = max(0, daily_pnl) / balance * 100
        if daily_profit_pct >= MAX_DAILY_PROFIT_PCT:
            return self._block(f"Daily profit {daily_profit_pct:.1f}% >= limit {MAX_DAILY_PROFIT_PCT}% (good day, stop trading)")

        # Stage 4: Position count
        positions = mt5f.get_bot_positions(MT5_MAGIC_NUMBER)
        if len(positions) >= MAX_CONCURRENT_POSITIONS:
            return self._block(f"Max positions reached: {len(positions)}/{MAX_CONCURRENT_POSITIONS}")

        # Stage 5: Daily trade count
        trade_count = mt5f.get_daily_trade_count(MT5_MAGIC_NUMBER)
        if trade_count >= MAX_DAILY_TRADES:
            return self._block(f"Max daily trades reached: {trade_count}/{MAX_DAILY_TRADES}")

        # Stage 6 (bonus): Total drawdown from peak
        if self.peak_balance > 0:
            dd_pct = (self.peak_balance - equity) / self.peak_balance * 100
            if dd_pct >= MAX_TOTAL_DD_PCT:
                return self._block(f"Total drawdown {dd_pct:.1f}% >= limit {MAX_TOTAL_DD_PCT}%")

        return {
            "allowed": True,
            "reason": "All checks passed",
            "lot_multiplier": self._lot_multiplier,
        }

    def check_pair_limit(self, symbol: str) -> bool:
        """Check if we can open another position for this pair."""
        from ..config import MAX_PER_PAIR_POSITIONS
        positions = mt5f.get_bot_positions(MT5_MAGIC_NUMBER)
        pair_count = sum(1 for p in positions if p["symbol"] == symbol)
        return pair_count < MAX_PER_PAIR_POSITIONS

    def update_streak(self, is_win: bool):
        """Update win/loss streak and adjust lot multiplier."""
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            if self.consecutive_wins >= CONSECUTIVE_WIN_INCREASE:
                self._lot_multiplier = min(1.5, self._lot_multiplier * 1.25)
                logger.info(f"Win streak {self.consecutive_wins}: lot multiplier -> {self._lot_multiplier:.2f}")
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            if self.consecutive_losses >= CONSECUTIVE_LOSS_REDUCE:
                self._lot_multiplier = max(0.25, self._lot_multiplier * 0.5)
                logger.info(f"Loss streak {self.consecutive_losses}: lot multiplier -> {self._lot_multiplier:.2f}")

    def get_adjusted_risk(self) -> float:
        """Get risk per trade adjusted by streak multiplier."""
        return RISK_PER_TRADE * self._lot_multiplier

    def _block(self, reason: str) -> dict:
        logger.warning(f"CIRCUIT BREAKER: {reason}")
        return {"allowed": False, "reason": reason, "lot_multiplier": self._lot_multiplier}
