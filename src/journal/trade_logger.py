"""
Trade Logger - Orchestrates Google Sheets + Notion journal writing
"""
import logging
from datetime import datetime, timezone
from .sheets_writer import SheetsWriter
from .notion_writer import NotionWriter

logger = logging.getLogger(__name__)


class TradeLogger:
    def __init__(self, sheets_creds: str = None, sheet_name: str = None,
                 notion_token: str = None, notion_db_id: str = None):
        self.sheets = None
        self.notion = None

        if sheets_creds and sheet_name:
            try:
                self.sheets = SheetsWriter(sheets_creds, sheet_name)
            except Exception as e:
                logger.error(f"Failed to init Sheets: {e}")

        if notion_token and notion_db_id:
            try:
                self.notion = NotionWriter(notion_token, notion_db_id)
            except Exception as e:
                logger.error(f"Failed to init Notion: {e}")

        self._open_trades = {}
        self._closed_trades = []

    def log_open(self, trade_data: dict) -> str:
        trade_id = trade_data.get("trade_id",
            f"{trade_data.get('symbol', 'UNK')}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")

        trade_data["trade_id"] = trade_id
        trade_data["datetime_open"] = trade_data.get("datetime_open",
            datetime.now(timezone.utc).isoformat())

        self._open_trades[trade_id] = trade_data.copy()
        logger.info(f"Trade opened: {trade_id}")
        return trade_id

    def log_close(self, trade_id: str, close_data: dict) -> dict:
        trade = self._open_trades.pop(trade_id, None)
        if not trade:
            logger.warning(f"Trade {trade_id} not found in open trades")
            trade = {"trade_id": trade_id}

        trade.update(close_data)
        trade["datetime_close"] = close_data.get("datetime_close",
            datetime.now(timezone.utc).isoformat())

        # Calculate duration
        try:
            open_time = datetime.fromisoformat(trade["datetime_open"])
            close_time = datetime.fromisoformat(trade["datetime_close"])
            trade["duration_minutes"] = int((close_time - open_time).total_seconds() / 60)
        except (KeyError, ValueError):
            pass

        # Write to both journals
        if self.sheets:
            try:
                self.sheets.append_trade(trade)
            except Exception as e:
                logger.error(f"Sheets write failed: {e}")

        if self.notion:
            try:
                self.notion.create_trade_page(trade)
            except Exception as e:
                logger.error(f"Notion write failed: {e}")

        self._closed_trades.append(trade)
        logger.info(f"Trade closed and journaled: {trade_id}, P/L: ${trade.get('pnl_usd', 0):.2f}")
        return trade

    def log_daily_summary(self, stats: dict):
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.sheets:
            self.sheets.update_daily_summary(date_str, stats)

    def log_weekly_review(self, review: dict, stats: dict):
        if self.notion:
            self.notion.create_weekly_review_page(review, stats)

    def get_closed_trades(self) -> list:
        return self._closed_trades.copy()

    def get_open_trades(self) -> dict:
        return self._open_trades.copy()
