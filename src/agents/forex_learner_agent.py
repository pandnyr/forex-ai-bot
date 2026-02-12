"""
Forex Learner Agent - Self-learning from trade outcomes
"Every loss = 1 prompt = 1 fix" (Sharbel approach)
"""
import json
import os
import logging
from datetime import datetime, timezone
from ..models.model_factory import ModelFactory
from ..config import BRAIN_MODEL, DATA_DIR, LEARNER_INTERVAL_TRADES

logger = logging.getLogger(__name__)

LOSS_ANALYSIS_PROMPT = """You are an expert forex trading analyst reviewing a losing trade.

LOSING TRADE:
{trade_data}

LAST 10 TRADES (for context):
{recent_trades}

MARKET CONDITIONS AT ENTRY:
{market_context}

Analyze this loss and provide actionable improvements. Respond in EXACTLY this JSON format:
{{
    "root_cause": "<single sentence: why did this trade lose?>",
    "filter_suggestion": "<what additional filter/condition would have prevented this trade?>",
    "parameter_change": "<which parameter should change and to what value? e.g. 'SCALP_RSI_THRESHOLD: 55' or 'none'>",
    "avoid_pattern": "<describe the pattern to avoid in future, or 'none'>",
    "severity": "<low|medium|high - how impactful is this learning?>"
}}
"""

WEEKLY_REVIEW_PROMPT = """You are reviewing a week of forex trading performance.

WEEKLY STATS:
{stats}

ALL LEARNINGS FROM THIS WEEK:
{learnings}

CURRENT STRATEGY PARAMETERS:
{parameters}

Provide a comprehensive weekly review. Respond in JSON:
{{
    "summary": "<3-5 sentence performance summary>",
    "top_3_issues": ["<issue1>", "<issue2>", "<issue3>"],
    "parameter_adjustments": {{"param_name": "new_value", ...}},
    "strategy_weights": {{"scalp_momentum": 0.5, "trend_following": 0.3, "mean_reversion": 0.1, "breakout": 0.1}},
    "focus_next_week": "<what to focus on improving>"
}}
"""


class ForexLearnerAgent:
    def __init__(self):
        self.model = ModelFactory()
        self.learnings = []
        self.learnings_file = os.path.join(DATA_DIR, "learnings.json")
        self._load_learnings()

    def _load_learnings(self):
        """Load accumulated learnings from disk."""
        try:
            if os.path.exists(self.learnings_file):
                with open(self.learnings_file, "r") as f:
                    self.learnings = json.load(f)
        except Exception as e:
            logger.error(f"Error loading learnings: {e}")
            self.learnings = []

    def _save_learnings(self):
        """Save learnings to disk."""
        os.makedirs(os.path.dirname(self.learnings_file), exist_ok=True)
        with open(self.learnings_file, "w") as f:
            json.dump(self.learnings, f, indent=2, default=str)

    def analyze_loss(self, trade_data: dict, recent_trades: list, market_context: str) -> dict:
        """Analyze a losing trade and extract a learning."""
        try:
            prompt = LOSS_ANALYSIS_PROMPT.format(
                trade_data=json.dumps(trade_data, indent=2, default=str),
                recent_trades=json.dumps(recent_trades[-10:], indent=2, default=str),
                market_context=market_context,
            )

            result = self.model.call(BRAIN_MODEL, prompt, json_mode=True)

            if isinstance(result, dict):
                learning = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "trade_id": trade_data.get("trade_id", "unknown"),
                    "symbol": trade_data.get("symbol", "unknown"),
                    "strategy": trade_data.get("strategy", "unknown"),
                    "pnl": trade_data.get("pnl_usd", 0),
                    **result,
                }
                self.learnings.append(learning)
                self._save_learnings()

                logger.info(f"Learning captured: {result.get('root_cause', 'N/A')}")
                return learning

            return {"error": "Failed to parse AI response"}

        except Exception as e:
            logger.error(f"Error analyzing loss: {e}")
            return {"error": str(e)}

    def generate_weekly_review(self, weekly_stats: dict, current_params: dict) -> dict:
        """Generate a comprehensive weekly performance review."""
        try:
            # Get learnings from this week
            week_learnings = self.learnings[-50:]  # Last 50 learnings

            prompt = WEEKLY_REVIEW_PROMPT.format(
                stats=json.dumps(weekly_stats, indent=2, default=str),
                learnings=json.dumps(week_learnings, indent=2, default=str),
                parameters=json.dumps(current_params, indent=2),
            )

            result = self.model.call(BRAIN_MODEL, prompt, json_mode=True, max_tokens=2048)

            if isinstance(result, dict):
                # Save weekly review
                review_file = os.path.join(DATA_DIR, f"weekly_review_{datetime.now().strftime('%Y%m%d')}.json")
                with open(review_file, "w") as f:
                    json.dump(result, f, indent=2)

                logger.info(f"Weekly review generated: {result.get('summary', '')[:100]}")
                return result

            return {"error": "Failed to parse weekly review"}

        except Exception as e:
            logger.error(f"Error generating weekly review: {e}")
            return {"error": str(e)}

    def get_active_filters(self) -> list:
        """Get accumulated avoid patterns for filtering signals."""
        high_severity = [l for l in self.learnings
                        if l.get("severity") == "high" and l.get("avoid_pattern", "none") != "none"]
        return [l["avoid_pattern"] for l in high_severity[-20:]]  # Last 20 high-severity patterns

    def get_learning_stats(self) -> dict:
        """Get summary of all learnings."""
        if not self.learnings:
            return {"total": 0}

        return {
            "total": len(self.learnings),
            "high_severity": sum(1 for l in self.learnings if l.get("severity") == "high"),
            "medium_severity": sum(1 for l in self.learnings if l.get("severity") == "medium"),
            "low_severity": sum(1 for l in self.learnings if l.get("severity") == "low"),
            "strategies_analyzed": list(set(l.get("strategy", "unknown") for l in self.learnings)),
        }
