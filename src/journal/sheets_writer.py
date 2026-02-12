"""
Google Sheets Trade Journal Writer
Uses gspread + google-auth for API access
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    logger.warning("gspread not available - Sheets logging disabled")


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

HEADER_ROW = [
    "Trade ID", "Open Time", "Close Time", "Symbol", "Direction",
    "Entry Price", "Exit Price", "SL", "TP", "Lot Size",
    "P/L ($)", "P/L (Pips)", "R-Multiple", "Strategy", "Session",
    "Regime", "AI Confidence", "Duration (min)", "Commission",
    "AI Analysis", "Learner Notes"
]


class SheetsWriter:
    def __init__(self, credentials_file: str, sheet_name: str):
        self.sheet = None
        self.sheet_name = sheet_name

        if not GSPREAD_AVAILABLE:
            logger.warning("Google Sheets not available")
            return

        try:
            creds = Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
            gc = gspread.authorize(creds)
            self.sheet = gc.open(sheet_name)
            self._ensure_headers()
            logger.info(f"Connected to Google Sheet: {sheet_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets: {e}")

    def _ensure_headers(self):
        try:
            ws = self._get_worksheet("Trades")
            first_row = ws.row_values(1)
            if not first_row or first_row[0] != HEADER_ROW[0]:
                ws.update("A1", [HEADER_ROW])
        except Exception as e:
            logger.error(f"Error setting headers: {e}")

    def _get_worksheet(self, name: str):
        try:
            return self.sheet.worksheet(name)
        except gspread.WorksheetNotFound:
            return self.sheet.add_worksheet(title=name, rows=1000, cols=25)

    def append_trade(self, trade: dict):
        if not self.sheet:
            return

        try:
            ws = self._get_worksheet("Trades")
            row = [
                trade.get("trade_id", ""),
                trade.get("datetime_open", ""),
                trade.get("datetime_close", ""),
                trade.get("symbol", ""),
                trade.get("direction", ""),
                trade.get("entry_price", ""),
                trade.get("exit_price", ""),
                trade.get("sl", ""),
                trade.get("tp", ""),
                trade.get("lot_size", ""),
                trade.get("pnl_usd", ""),
                trade.get("pnl_pips", ""),
                trade.get("r_multiple", ""),
                trade.get("strategy", ""),
                trade.get("session", ""),
                trade.get("regime", ""),
                trade.get("ai_confidence", ""),
                trade.get("duration_minutes", ""),
                trade.get("commission", ""),
                trade.get("ai_analysis", ""),
                trade.get("learner_notes", ""),
            ]
            ws.append_row(row, value_input_option="USER_ENTERED")
            logger.info(f"Trade logged to Sheets: {trade.get('trade_id', 'N/A')}")
        except Exception as e:
            logger.error(f"Error appending trade to Sheets: {e}")

    def update_daily_summary(self, date: str, stats: dict):
        if not self.sheet:
            return

        try:
            ws = self._get_worksheet("Daily Summary")
            headers = ws.row_values(1)
            if not headers:
                ws.update("A1", [["Date", "Trades", "Winners", "Losers", "Win Rate",
                                   "Total P/L", "Best Trade", "Worst Trade", "Max DD"]])

            row = [
                date,
                stats.get("total_trades", 0),
                stats.get("winners", 0),
                stats.get("losers", 0),
                f"{stats.get('win_rate', 0):.1f}%",
                f"${stats.get('total_pnl', 0):.2f}",
                f"${stats.get('best_trade', 0):.2f}",
                f"${stats.get('worst_trade', 0):.2f}",
                f"{stats.get('max_dd', 0):.1f}%",
            ]
            ws.append_row(row, value_input_option="USER_ENTERED")
        except Exception as e:
            logger.error(f"Error updating daily summary: {e}")

    def update_strategy_stats(self, strategy_name: str, stats: dict):
        if not self.sheet:
            return

        try:
            ws = self._get_worksheet("Strategy Stats")
            headers = ws.row_values(1)
            if not headers:
                ws.update("A1", [["Strategy", "Total Trades", "Win Rate", "Profit Factor",
                                   "Avg R-Multiple", "Total P/L", "Last Updated"]])

            row = [
                strategy_name,
                stats.get("total_trades", 0),
                f"{stats.get('win_rate', 0):.1f}%",
                f"{stats.get('profit_factor', 0):.2f}",
                f"{stats.get('avg_r_multiple', 0):.2f}",
                f"${stats.get('total_pnl', 0):.2f}",
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            ]
            ws.append_row(row, value_input_option="USER_ENTERED")
        except Exception as e:
            logger.error(f"Error updating strategy stats: {e}")
