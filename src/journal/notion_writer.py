"""
Notion Trade Journal Writer
Each trade = 1 Notion page in a database
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    from notion_client import Client as NotionClient
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False
    logger.warning("notion-client not available - Notion logging disabled")


class NotionWriter:
    def __init__(self, token: str, database_id: str):
        self.client = None
        self.database_id = database_id

        if not NOTION_AVAILABLE:
            logger.warning("Notion not available")
            return

        if not token or not database_id:
            logger.warning("Notion token or database_id not configured")
            return

        try:
            self.client = NotionClient(auth=token)
            logger.info("Connected to Notion API")
        except Exception as e:
            logger.error(f"Failed to connect to Notion: {e}")

    def create_trade_page(self, trade: dict):
        if not self.client:
            return

        try:
            properties = {
                "Trade ID": {"title": [{"text": {"content": trade.get("trade_id", "Unknown")}}]},
                "Symbol": {"select": {"name": trade.get("symbol", "N/A")}},
                "Direction": {"select": {"name": trade.get("direction", "N/A").upper()}},
                "Strategy": {"select": {"name": trade.get("strategy", "N/A")}},
                "Session": {"select": {"name": trade.get("session", "N/A")}},
                "Regime": {"select": {"name": trade.get("regime", "N/A")}},
            }

            if trade.get("entry_price"):
                properties["Entry Price"] = {"number": float(trade["entry_price"])}
            if trade.get("exit_price"):
                properties["Exit Price"] = {"number": float(trade["exit_price"])}
            if trade.get("sl"):
                properties["Stop Loss"] = {"number": float(trade["sl"])}
            if trade.get("tp"):
                properties["Take Profit"] = {"number": float(trade["tp"])}
            if trade.get("lot_size"):
                properties["Lot Size"] = {"number": float(trade["lot_size"])}
            if trade.get("pnl_usd") is not None:
                properties["P/L ($)"] = {"number": float(trade["pnl_usd"])}
            if trade.get("pnl_pips") is not None:
                properties["P/L (Pips)"] = {"number": float(trade["pnl_pips"])}
            if trade.get("r_multiple") is not None:
                properties["R-Multiple"] = {"number": float(trade["r_multiple"])}
            if trade.get("ai_confidence") is not None:
                properties["AI Confidence"] = {"number": int(trade["ai_confidence"])}
            if trade.get("duration_minutes"):
                properties["Duration (min)"] = {"number": int(trade["duration_minutes"])}
            if trade.get("datetime_open"):
                properties["Open Time"] = {"date": {"start": trade["datetime_open"]}}
            if trade.get("datetime_close"):
                properties["Close Time"] = {"date": {"start": trade["datetime_close"]}}

            children = []
            if trade.get("ai_analysis"):
                children.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "AI Analysis"}}]},
                })
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": str(trade["ai_analysis"])[:2000]}}]},
                })

            if trade.get("learner_notes"):
                children.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "Learner Notes"}}]},
                })
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": str(trade["learner_notes"])[:2000]}}]},
                })

            page = self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=properties,
                children=children if children else None,
            )

            logger.info(f"Trade page created in Notion: {trade.get('trade_id', 'N/A')}")
            return page.get("id")

        except Exception as e:
            logger.error(f"Error creating Notion page: {e}")
            return None

    def create_weekly_review_page(self, review: dict, stats: dict):
        if not self.client:
            return

        try:
            title = f"Weekly Review - {datetime.now(timezone.utc).strftime('%Y-W%V')}"

            children = [
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "Performance Summary"}}]},
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": review.get("summary", "No summary")}}]},
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "Stats"}}]},
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content":
                        f"Trades: {stats.get('total_trades', 0)} | "
                        f"Win Rate: {stats.get('win_rate', 0):.1f}% | "
                        f"P/L: ${stats.get('total_pnl', 0):.2f} | "
                        f"PF: {stats.get('profit_factor', 0):.2f}"
                    }}]},
                },
            ]

            if review.get("top_3_issues"):
                children.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "Top Issues"}}]},
                })
                for issue in review["top_3_issues"]:
                    children.append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"text": {"content": issue}}]},
                    })

            if review.get("focus_next_week"):
                children.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "Focus Next Week"}}]},
                })
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": review["focus_next_week"]}}]},
                })

            self.client.pages.create(
                parent={"database_id": self.database_id},
                properties={
                    "Trade ID": {"title": [{"text": {"content": title}}]},
                },
                children=children,
            )

            logger.info(f"Weekly review created in Notion: {title}")

        except Exception as e:
            logger.error(f"Error creating weekly review in Notion: {e}")
