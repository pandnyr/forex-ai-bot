"""
Forex Journal Agent - Automated trade journaling
Logs to Google Sheets + Notion in parallel
"""
import logging
from datetime import datetime, timezone
from ..config import JOURNAL_COLUMNS

logger = logging.getLogger(__name__)


class ForexJournalAgent:
    def __init__(self, sheets_writer=None, notion_writer=None):
        self.sheets = sheets_writer
        self.notion = notion_writer
        self.trades = []

    def log_trade_open(self, trade_data: dict):
        """Log a trade opening."""
        trade = {col: trade_data.get(col, "") for col in JOURNAL_COLUMNS}
        trade["trade_id"] = f"{trade_data['symbol']}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        trade["datetime_open"] = datetime.now(timezone.utc).isoformat()
        self.trades.append(trade)
        logger.info(f"Trade opened: {trade['trade_id']}")
        return trade["trade_id"]

    def log_trade_close(self, trade_id: str, close_data: dict):
        """Log a trade closing and write to journals."""
        for trade in self.trades:
            if trade["trade_id"] == trade_id:
                trade.update(close_data)
                trade["datetime_close"] = datetime.now(timezone.utc).isoformat()

                # Calculate R-multiple
                if trade.get("sl") and trade.get("entry_price"):
                    risk = abs(trade["entry_price"] - trade["sl"])
                    if risk > 0 and trade.get("pnl_pips"):
                        trade["r_multiple"] = round(trade["pnl_pips"] / (risk * 10000), 2)

                # Write to journals
                self._write_to_sheets(trade)
                self._write_to_notion(trade)

                logger.info(f"Trade closed: {trade_id}, P/L: ${trade.get('pnl_usd', 0):.2f}")
                return trade

        logger.warning(f"Trade {trade_id} not found")
        return None

    def _write_to_sheets(self, trade: dict):
        """Write trade to Google Sheets."""
        if self.sheets:
            try:
                self.sheets.append_trade(trade)
            except Exception as e:
                logger.error(f"Error writing to Sheets: {e}")

    def _write_to_notion(self, trade: dict):
        """Write trade to Notion."""
        if self.notion:
            try:
                self.notion.create_trade_page(trade)
            except Exception as e:
                logger.error(f"Error writing to Notion: {e}")

    def get_trade_stats(self, last_n: int = None) -> dict:
        """Calculate performance statistics."""
        trades = self.trades if last_n is None else self.trades[-last_n:]
        closed = [t for t in trades if t.get("datetime_close")]

        if not closed:
            return {"total": 0}

        pnls = [t.get("pnl_usd", 0) for t in closed]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        win_rate = len(winners) / len(pnls) * 100 if pnls else 0
        avg_win = sum(winners) / len(winners) if winners else 0
        avg_loss = sum(losers) / len(losers) if losers else 0
        profit_factor = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf')

        return {
            "total_trades": len(closed),
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "best_trade": round(max(pnls), 2) if pnls else 0,
            "worst_trade": round(min(pnls), 2) if pnls else 0,
            "consecutive_wins": self._max_streak(pnls, positive=True),
            "consecutive_losses": self._max_streak(pnls, positive=False),
        }

    def _max_streak(self, pnls: list, positive: bool) -> int:
        max_s = current = 0
        for p in pnls:
            if (positive and p > 0) or (not positive and p < 0):
                current += 1
                max_s = max(max_s, current)
            else:
                current = 0
        return max_s
