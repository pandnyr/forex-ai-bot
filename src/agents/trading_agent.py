"""
Forex Trading Agent - AI-powered trade decisions
Uses Brain+Muscles pattern: rule-based signal -> AI confirmation -> rule-based execution
"""
import logging
from datetime import datetime, timezone
from ..models.model_factory import ModelFactory
from ..config import (MUSCLE_MODEL, SCALP_MIN_AI_CONFIDENCE, AI_MAX_TOKENS,
                       AI_TEMPERATURE, MT5_MAGIC_NUMBER)
from .. import nice_funcs_mt5 as mt5f

logger = logging.getLogger(__name__)
model = ModelFactory()

TRADE_ANALYSIS_PROMPT = """You are a forex trading analyst. Analyze this trade signal and provide your assessment.

Signal: {direction} {symbol}
Strategy: {strategy}
Reasoning: {reasoning}

Current Market Data (last 20 bars on entry timeframe):
{market_data}

Higher Timeframe Context:
{htf_context}

Current Spread: {spread} pips
Account Balance: ${balance}
Open Positions: {open_positions}

Evaluate this trade on a scale of 0-100:
- Is the trend alignment strong?
- Are there any conflicting signals?
- Is the risk/reward acceptable?
- Any upcoming high-impact news concerns?

Respond in EXACTLY this JSON format:
{{"confidence": <0-100>, "reasoning": "<brief explanation>", "adjust_sl": <null or pips>, "adjust_tp": <null or pips>}}
"""


class TradingAgent:
    def __init__(self):
        self.model = ModelFactory()
        self.trade_count = 0

    def evaluate_signal(self, signal: dict, bars_dict: dict) -> dict:
        """
        Evaluate a strategy signal using AI.
        signal: {"symbol", "direction", "sl_pips", "tp_pips", "confidence", "strategy", "reasoning"}
        bars_dict: {"M5": df, "M15": df, "H1": df, "H4": df}
        Returns: {"approved": bool, "confidence": int, "reasoning": str, "sl_pips": float, "tp_pips": float}
        """
        try:
            # Prepare market data summary
            m5 = bars_dict.get("M5")
            h1 = bars_dict.get("H1")
            market_data = ""
            if m5 is not None and len(m5) > 0:
                last20 = m5.tail(20)
                market_data = last20.to_string()

            htf_context = ""
            if h1 is not None and len(h1) > 0:
                last10 = h1.tail(10)
                htf_context = last10.to_string()

            spread = mt5f.get_spread(signal["symbol"])
            balance = mt5f.get_account_balance()
            positions = mt5f.get_bot_positions(MT5_MAGIC_NUMBER)

            prompt = TRADE_ANALYSIS_PROMPT.format(
                direction=signal["direction"],
                symbol=signal["symbol"],
                strategy=signal["strategy"],
                reasoning=signal["reasoning"],
                market_data=market_data,
                htf_context=htf_context,
                spread=f"{spread:.1f}",
                balance=f"{balance:.2f}",
                open_positions=len(positions),
            )

            result = self.model.call(
                MUSCLE_MODEL, prompt,
                max_tokens=AI_MAX_TOKENS,
                temperature=AI_TEMPERATURE,
                json_mode=True
            )

            if isinstance(result, dict):
                confidence = result.get("confidence", 0)
                reasoning = result.get("reasoning", "")
                sl = result.get("adjust_sl") or signal["sl_pips"]
                tp = result.get("adjust_tp") or signal["tp_pips"]

                approved = confidence >= SCALP_MIN_AI_CONFIDENCE

                logger.info(f"AI evaluation: {signal['symbol']} {signal['direction']} - "
                           f"Confidence: {confidence}, Approved: {approved}")

                return {
                    "approved": approved,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "sl_pips": sl,
                    "tp_pips": tp,
                }
            else:
                logger.warning(f"AI returned non-JSON response: {str(result)[:200]}")
                return {
                    "approved": False,
                    "confidence": 0,
                    "reasoning": "Failed to parse AI response",
                    "sl_pips": signal["sl_pips"],
                    "tp_pips": signal["tp_pips"],
                }

        except Exception as e:
            logger.error(f"Error in AI evaluation: {e}")
            return {
                "approved": False,
                "confidence": 0,
                "reasoning": f"Error: {str(e)}",
                "sl_pips": signal["sl_pips"],
                "tp_pips": signal["tp_pips"],
            }

    def execute_trade(self, signal: dict, ai_result: dict) -> dict:
        """Execute a trade based on approved signal."""
        try:
            from ..config import RISK_PER_TRADE, COMPOUND_ENABLED

            balance = mt5f.get_account_balance()
            risk_usd = balance * (RISK_PER_TRADE / 100)

            sl_pips = ai_result["sl_pips"]
            tp_pips = ai_result["tp_pips"]

            comment = f"{signal['strategy']}|{ai_result['confidence']}"

            if signal["direction"] == "buy":
                result = mt5f.market_buy(
                    signal["symbol"], risk_usd, sl_pips, tp_pips, comment=comment
                )
            else:
                result = mt5f.market_sell(
                    signal["symbol"], risk_usd, sl_pips, tp_pips, comment=comment
                )

            if result["success"]:
                self.trade_count += 1
                logger.info(f"Trade executed: {signal['direction'].upper()} {signal['symbol']} "
                           f"Ticket: {result['ticket']}")
            else:
                logger.error(f"Trade execution failed: {result.get('error', 'Unknown')}")

            return result

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {"success": False, "error": str(e)}

    def manage_open_positions(self):
        """Manage trailing stops on open positions."""
        from ..config import MT5_MAGIC_NUMBER
        positions = mt5f.get_bot_positions(MT5_MAGIC_NUMBER)

        for pos in positions:
            # Trail stop at 50% of current profit
            if pos["profit"] > 0:
                symbol_info = mt5f.get_symbol_info(pos["symbol"])
                if not symbol_info:
                    continue
                digits = symbol_info["digits"]
                point = symbol_info["point"]
                pip_size = point * 10 if (digits == 5 or digits == 3) else point

                # Calculate profit in pips
                if pos["type"] == "buy":
                    profit_pips = (pos["price_current"] - pos["price_open"]) / pip_size
                else:
                    profit_pips = (pos["price_open"] - pos["price_current"]) / pip_size

                # Start trailing after 10 pips profit
                if profit_pips > 10:
                    trail_pips = max(8, profit_pips * 0.5)
                    mt5f.trail_stop(pos["ticket"], trail_pips)
