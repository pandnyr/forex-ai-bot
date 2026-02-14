"""
Forex AI Trading Bot - Main Loop
Brain + Muscles Pattern: Rule-based signals -> AI confirmation -> Rule-based execution
"""
import sys
import os
import time
import logging
import json
from datetime import datetime, timezone

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"logs/bot_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")

from .config import (
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH,
    MT5_SYMBOLS, MT5_MAGIC_NUMBER,
    LOOP_INTERVAL_SECONDS, LEARNER_INTERVAL_TRADES,
    GOOGLE_SHEETS_CREDS, GOOGLE_SHEET_NAME,
    NOTION_TOKEN, NOTION_DATABASE_ID,
    COMPOUND_ENABLED, RISK_PER_TRADE,
    get_strategy_config,
)
from . import nice_funcs_mt5 as mt5f
from .agents.trading_agent import TradingAgent
from .agents.risk_agent import RiskAgent
from .agents.forex_session_agent import ForexSessionAgent
from .agents.forex_scanner_agent import ForexScannerAgent
from .agents.forex_learner_agent import ForexLearnerAgent
from .agents.forex_journal_agent import ForexJournalAgent
from .strategies.multi_strategy import MultiStrategyOrchestrator
from .journal.trade_logger import TradeLogger
from .models.model_factory import ModelFactory


class ForexBot:
    def __init__(self):
        logger.info("=" * 60)
        logger.info("Forex AI Trading Bot - Initializing")
        logger.info("=" * 60)

        # Initialize MT5
        if not mt5f.initialize(
            path=MT5_PATH or None,
            login=MT5_LOGIN or None,
            password=MT5_PASSWORD or None,
            server=MT5_SERVER or None,
        ):
            logger.error("Failed to initialize MT5. Exiting.")
            sys.exit(1)

        account = mt5f.get_account_info()
        logger.info(f"Account: {account.get('server', 'N/A')}, "
                     f"Balance: ${account.get('balance', 0):.2f}, "
                     f"Leverage: 1:{account.get('leverage', 0)}")

        # Initialize components
        self.risk_agent = RiskAgent()
        self.risk_agent.peak_balance = account.get("balance", 0)

        self.session_agent = ForexSessionAgent()
        self.scanner = ForexScannerAgent()
        self.trading_agent = TradingAgent()
        self.learner = ForexLearnerAgent()
        self.journal_agent = ForexJournalAgent()

        # Strategy orchestrator
        self.strategy = MultiStrategyOrchestrator(get_strategy_config())

        # Trade logger (Sheets + Notion)
        self.trade_logger = TradeLogger(
            sheets_creds=GOOGLE_SHEETS_CREDS if os.path.exists(GOOGLE_SHEETS_CREDS or "") else None,
            sheet_name=GOOGLE_SHEET_NAME,
            notion_token=NOTION_TOKEN or None,
            notion_db_id=NOTION_DATABASE_ID or None,
        )
        self.journal_agent.sheets = self.trade_logger.sheets
        self.journal_agent.notion = self.trade_logger.notion

        # State tracking
        self._active_trade_ids = {}  # ticket -> trade_id mapping
        self._total_trades = 0
        self._previous_positions = set()

        logger.info("All components initialized successfully")

    def run(self):
        """Main bot loop."""
        logger.info("Starting main trading loop...")

        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

            time.sleep(LOOP_INTERVAL_SECONDS)

        # Cleanup
        self._shutdown()

    def _tick(self):
        """Single iteration of the main loop."""

        # 1. Session check
        session_info = self.session_agent.get_current_session()
        if not session_info["active"]:
            if session_info["session"] == "weekend":
                logger.info("Market closed (weekend). Sleeping 5 minutes...")
                time.sleep(300)
                return
            # Off-hours: only manage positions
            self._manage_positions()
            return

        # 2. Risk circuit breaker
        risk_check = self.risk_agent.check_all()
        if not risk_check["allowed"]:
            logger.warning(f"Circuit breaker active: {risk_check['reason']}")
            self._manage_positions()  # Still manage open positions
            return

        # 3. Check for closed trades (detect exits)
        self._detect_closed_trades(session_info)

        # 4. Manage existing positions (trailing stops)
        self._manage_positions()

        # 5. Scan for new signals
        active_symbols = session_info["symbols"]
        if not active_symbols:
            return

        # Get market data for active symbols
        market_data = self.scanner.scan_all_pairs(active_symbols)

        for symbol in active_symbols:
            bars_dict = market_data.get(symbol)
            if not bars_dict:
                continue

            # Check if we can trade this pair
            if not self.risk_agent.check_pair_limit(symbol):
                continue

            # Get market overview for regime detection
            overview = self.scanner.get_market_overview([symbol])
            regime = overview[0]["regime"] if overview else "transitional"

            # Get signals from strategy orchestrator
            signals = self.strategy.get_signals(symbol, bars_dict, session_info["session"])

            for signal in signals:
                # AI confirmation (Muscle model - fast + cheap)
                ai_result = self.trading_agent.evaluate_signal(signal, bars_dict)

                if not ai_result["approved"]:
                    logger.info(f"Signal rejected by AI: {symbol} {signal['direction']} "
                               f"(confidence: {ai_result['confidence']})")
                    continue

                # Final risk check before execution
                risk_recheck = self.risk_agent.check_all()
                if not risk_recheck["allowed"]:
                    logger.warning(f"Risk check failed before execution: {risk_recheck['reason']}")
                    break

                # Execute trade
                result = self.trading_agent.execute_trade(signal, ai_result)

                if result.get("success"):
                    # Log to journal
                    trade_data = {
                        "symbol": symbol,
                        "direction": signal["direction"],
                        "entry_price": result.get("price", 0),
                        "sl": 0,  # Will be filled by MT5
                        "tp": 0,
                        "lot_size": result.get("volume", 0),
                        "strategy": signal["strategy"],
                        "session": session_info["session"],
                        "regime": regime,
                        "ai_confidence": ai_result["confidence"],
                        "ai_analysis": ai_result["reasoning"],
                    }
                    trade_id = self.journal_agent.log_trade_open(trade_data)
                    self._active_trade_ids[result["ticket"]] = trade_id
                    self._total_trades += 1

                    logger.info(f"TRADE OPENED: {signal['direction'].upper()} {symbol} "
                               f"@ {result.get('price', 0):.5f} | "
                               f"AI Confidence: {ai_result['confidence']} | "
                               f"Strategy: {signal['strategy']}")

                    # Only one trade per symbol per tick
                    break

    def _manage_positions(self):
        """Manage trailing stops on open positions."""
        self.trading_agent.manage_open_positions()

    def _detect_closed_trades(self, session_info: dict):
        """Detect trades that have been closed (SL/TP hit) and log them."""
        current_positions = set()
        for pos in mt5f.get_bot_positions(MT5_MAGIC_NUMBER):
            current_positions.add(pos["ticket"])

        # Find closed positions
        closed_tickets = self._previous_positions - current_positions
        for ticket in closed_tickets:
            trade_id = self._active_trade_ids.pop(ticket, None)
            if not trade_id:
                continue

            # Get trade result from history
            history = mt5f.get_trade_history(days=1)
            trade_pnl = 0.0
            exit_price = 0.0
            if not history.empty:
                ticket_deals = history[history.get("position_id", history.get("order", 0)) == ticket]
                if not ticket_deals.empty:
                    trade_pnl = ticket_deals["profit"].sum()
                    exit_price = ticket_deals.iloc[-1].get("price", 0)

            close_data = {
                "exit_price": exit_price,
                "pnl_usd": trade_pnl,
                "pnl_pips": 0,  # Will be calculated by journal
            }

            trade = self.journal_agent.log_trade_close(trade_id, close_data)
            self.trade_logger.log_close(trade_id, close_data)

            # Update risk agent streak
            is_win = trade_pnl > 0
            self.risk_agent.update_streak(is_win)

            # If loss, trigger learner agent
            if not is_win and trade:
                recent = self.journal_agent.trades[-10:]
                overview = self.scanner.get_market_overview()
                market_ctx = json.dumps(overview, default=str) if overview else "N/A"
                self.learner.analyze_loss(trade, recent, market_ctx)

            # Periodic cumulative learning
            if self._total_trades > 0 and self._total_trades % LEARNER_INTERVAL_TRADES == 0:
                stats = self.journal_agent.get_trade_stats()
                self.learner.generate_weekly_review(stats, {})

            logger.info(f"TRADE CLOSED: {trade_id} | P/L: ${trade_pnl:.2f} | "
                       f"{'WIN' if is_win else 'LOSS'}")

        self._previous_positions = current_positions

    def _shutdown(self):
        """Clean shutdown."""
        # Log final stats
        stats = self.journal_agent.get_trade_stats()
        if stats.get("total_trades", 0) > 0:
            logger.info(f"Session stats: {json.dumps(stats, indent=2)}")
            self.trade_logger.log_daily_summary(stats)

        # Log AI cost
        model = ModelFactory()
        usage = model.usage_summary
        logger.info(f"AI usage: {json.dumps(usage, indent=2)}")

        # Shutdown MT5
        mt5f.shutdown()
        logger.info("Bot shutdown complete")


def main():
    bot = ForexBot()
    bot.run()


if __name__ == "__main__":
    main()
