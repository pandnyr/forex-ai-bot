"""
Forex AI Trading Bot - Entry Point
Run this file to start the bot.

Usage:
    python run.py              # Start live/demo trading
    python run.py --backtest   # Run backtests
    python run.py --status     # Show account status
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--backtest":
            run_backtest()
        elif sys.argv[1] == "--status":
            show_status()
        elif sys.argv[1] == "--help":
            print(__doc__)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print(__doc__)
    else:
        run_live()


def run_live():
    """Start the live trading bot."""
    from src.main import ForexBot
    bot = ForexBot()
    bot.run()


def run_backtest():
    """Run backtests on all strategies."""
    from datetime import datetime, timezone
    from src.backtester.engine import BacktestEngine
    from src.backtester.data_loader import DataLoader
    from src.strategies.scalp_momentum import ScalpMomentumStrategy
    from src.strategies.trend_following import TrendFollowingStrategy
    from src.strategies.mean_reversion import MeanReversionStrategy
    from src.strategies.breakout import BreakoutStrategy
    from src.config import (
        BACKTEST_INITIAL_BALANCE, BACKTEST_COMMISSION_PER_LOT,
        BACKTEST_SLIPPAGE_PIPS, get_strategy_config,
    )

    loader = DataLoader()
    engine = BacktestEngine(
        initial_balance=BACKTEST_INITIAL_BALANCE,
        commission_per_lot=BACKTEST_COMMISSION_PER_LOT,
        slippage_pips=BACKTEST_SLIPPAGE_PIPS,
    )

    config = get_strategy_config()
    strategies = [
        ("Scalp Momentum", ScalpMomentumStrategy(config)),
        ("Trend Following", TrendFollowingStrategy(config)),
        ("Mean Reversion", MeanReversionStrategy(config)),
        ("Breakout", BreakoutStrategy(config)),
    ]

    symbol = "EURUSD"
    print(f"\nRunning backtests for {symbol}...")
    print("=" * 60)

    # Try to load data
    try:
        start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bars_dict = loader.prepare_multi_tf(symbol, start, end, base_tf="M5")
    except FileNotFoundError:
        print("No historical data found.")
        print("Please place CSV data files in src/data/historical/")
        print("  Expected format: EURUSD_M5_YYYYMMDD_YYYYMMDD.csv")
        print("  Columns: time, open, high, low, close, volume")
        print("\nOr connect to MT5 on Windows to download data automatically.")
        return

    for name, strategy in strategies:
        print(f"\nTesting: {name}")
        print("-" * 40)
        result = engine.run(strategy, bars_dict, symbol)
        engine.print_summary(result)

        if result.meets_criteria():
            print(f"  -> PASSED minimum criteria")
        else:
            print(f"  -> FAILED minimum criteria")


def show_status():
    """Show current account and bot status."""
    from src import nice_funcs_mt5 as mt5f
    from src.config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH, MT5_MAGIC_NUMBER

    if not mt5f.initialize(
        path=MT5_PATH or None,
        login=MT5_LOGIN or None,
        password=MT5_PASSWORD or None,
        server=MT5_SERVER or None,
    ):
        print("Failed to connect to MT5")
        return

    account = mt5f.get_account_info()
    print(f"\nAccount Status")
    print("=" * 40)
    print(f"Server:      {account.get('server', 'N/A')}")
    print(f"Balance:     ${account.get('balance', 0):,.2f}")
    print(f"Equity:      ${account.get('equity', 0):,.2f}")
    print(f"Free Margin: ${account.get('free_margin', 0):,.2f}")
    print(f"Leverage:    1:{account.get('leverage', 0)}")
    print(f"Profit:      ${account.get('profit', 0):,.2f}")

    positions = mt5f.get_bot_positions(MT5_MAGIC_NUMBER)
    print(f"\nOpen Positions: {len(positions)}")
    for pos in positions:
        print(f"  {pos['symbol']} {pos['type'].upper()} {pos['volume']} lot "
              f"@ {pos['price_open']:.5f} -> {pos['price_current']:.5f} "
              f"P/L: ${pos['profit']:.2f}")

    daily_pnl = mt5f.get_daily_pnl(MT5_MAGIC_NUMBER)
    daily_trades = mt5f.get_daily_trade_count(MT5_MAGIC_NUMBER)
    print(f"\nToday: {daily_trades} trades, P/L: ${daily_pnl:.2f}")

    mt5f.shutdown()


if __name__ == "__main__":
    main()
