"""
Backtest Engine - Bar-by-bar simulation with realistic costs
"""
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    trade_id: str
    symbol: str
    direction: str  # "buy" or "sell"
    entry_price: float
    entry_time: datetime
    lot_size: float
    sl: float
    tp: float
    strategy: str
    confidence: int = 0
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    pnl_usd: float = 0.0
    pnl_pips: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    initial_balance: float = 10000.0
    final_balance: float = 0.0
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    total_pnl: float = 0.0
    avg_r_multiple: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0

    def meets_criteria(self, min_sharpe=1.0, min_pf=1.5, min_wr=0.50,
                       max_dd=20.0, min_trades=200) -> bool:
        return (self.sharpe_ratio >= min_sharpe and
                self.profit_factor >= min_pf and
                self.win_rate >= min_wr and
                self.max_drawdown_pct <= max_dd and
                self.total_trades >= min_trades)


class BacktestEngine:
    def __init__(self, initial_balance: float = 10000.0,
                 commission_per_lot: float = 7.0,
                 slippage_pips: float = 0.5,
                 risk_per_trade_pct: float = 1.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
        self.risk_per_trade_pct = risk_per_trade_pct

        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.equity_curve = []
        self.peak_balance = initial_balance
        self.max_dd = 0.0
        self._trade_counter = 0

    def reset(self):
        """Reset engine state for a new backtest run."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.open_trades = []
        self.closed_trades = []
        self.equity_curve = []
        self.peak_balance = self.initial_balance
        self.max_dd = 0.0
        self._trade_counter = 0

    def run(self, strategy, bars_dict: dict, symbol: str = "EURUSD",
            digits: int = 5) -> BacktestResult:
        """
        Run a backtest with given strategy on bar data.
        bars_dict should have the primary timeframe bars as a DataFrame with OHLCV.
        The strategy.analyze() is called on each bar.
        """
        self.reset()

        pip_size = 0.0001 if digits in (4, 5) else 0.01
        point = 0.00001 if digits == 5 else (0.001 if digits == 3 else pip_size)

        # Get primary timeframe (smallest)
        primary_tf = None
        primary_df = None
        for tf in ["M5", "M15", "M30", "H1"]:
            if tf in bars_dict and len(bars_dict[tf]) > 0:
                primary_tf = tf
                primary_df = bars_dict[tf]
                break

        if primary_df is None:
            logger.error("No valid bar data provided")
            return self._compile_results()

        logger.info(f"Starting backtest: {symbol}, {len(primary_df)} bars on {primary_tf}")

        for i in range(50, len(primary_df)):  # Start at 50 for indicator warmup
            bar = primary_df.iloc[i]
            bar_time = primary_df.index[i]

            # Build lookback window for strategy
            window = {}
            for tf, df in bars_dict.items():
                mask = df.index <= bar_time
                window[tf] = df[mask].tail(200)

            # 1. Check open trades against current bar (SL/TP hits)
            self._check_exits(bar, bar_time, pip_size, digits)

            # 2. Update equity
            unrealized = sum(self._calc_unrealized(t, bar["close"], pip_size) for t in self.open_trades)
            self.equity = self.balance + unrealized
            self.equity_curve.append({"time": bar_time, "equity": self.equity, "balance": self.balance})

            # Track drawdown
            if self.equity > self.peak_balance:
                self.peak_balance = self.equity
            dd = (self.peak_balance - self.equity) / self.peak_balance * 100
            self.max_dd = max(self.max_dd, dd)

            # 3. Generate signal from strategy
            if len(self.open_trades) < 3:  # Max concurrent
                signal = strategy.analyze(symbol, window)
                if signal:
                    self._open_trade(signal, bar, bar_time, pip_size, digits)

        # Close any remaining open trades at last bar close
        last_bar = primary_df.iloc[-1]
        last_time = primary_df.index[-1]
        for trade in list(self.open_trades):
            self._close_trade(trade, last_bar["close"], last_time, "end_of_data", pip_size)

        return self._compile_results()

    def _open_trade(self, signal: dict, bar, bar_time, pip_size: float, digits: int):
        """Open a new trade based on signal."""
        self._trade_counter += 1

        direction = signal["direction"]
        sl_pips = signal["sl_pips"]
        tp_pips = signal["tp_pips"]

        # Calculate lot size
        risk_usd = self.balance * (self.risk_per_trade_pct / 100)
        pip_value_per_lot = 10.0  # Approximate for major pairs
        lot = risk_usd / (sl_pips * pip_value_per_lot)
        lot = max(0.01, round(lot, 2))

        # Entry price with slippage
        slippage = self.slippage_pips * pip_size
        if direction == "buy":
            entry = bar["close"] + slippage
            sl = round(entry - sl_pips * pip_size, digits)
            tp = round(entry + tp_pips * pip_size, digits)
        else:
            entry = bar["close"] - slippage
            sl = round(entry + sl_pips * pip_size, digits)
            tp = round(entry - tp_pips * pip_size, digits)

        # Commission
        commission = self.commission_per_lot * lot
        self.balance -= commission

        trade = Trade(
            trade_id=f"BT-{self._trade_counter:04d}",
            symbol=signal["symbol"],
            direction=direction,
            entry_price=entry,
            entry_time=bar_time,
            lot_size=lot,
            sl=sl,
            tp=tp,
            strategy=signal.get("strategy", "unknown"),
            confidence=signal.get("confidence", 0),
            commission=commission,
        )
        self.open_trades.append(trade)

    def _check_exits(self, bar, bar_time, pip_size: float, digits: int):
        """Check if any open trades hit SL or TP."""
        for trade in list(self.open_trades):
            exit_price = None
            exit_reason = ""

            if trade.direction == "buy":
                # Check SL (low <= sl)
                if bar["low"] <= trade.sl:
                    exit_price = trade.sl
                    exit_reason = "sl_hit"
                # Check TP (high >= tp)
                elif bar["high"] >= trade.tp:
                    exit_price = trade.tp
                    exit_reason = "tp_hit"
            else:  # sell
                # Check SL (high >= sl)
                if bar["high"] >= trade.sl:
                    exit_price = trade.sl
                    exit_reason = "sl_hit"
                # Check TP (low <= tp)
                elif bar["low"] <= trade.tp:
                    exit_price = trade.tp
                    exit_reason = "tp_hit"

            if exit_price:
                self._close_trade(trade, exit_price, bar_time, exit_reason, pip_size)

    def _close_trade(self, trade: Trade, exit_price: float, exit_time, reason: str, pip_size: float):
        """Close a trade and calculate P/L."""
        if trade.direction == "buy":
            pnl_pips = (exit_price - trade.entry_price) / pip_size
        else:
            pnl_pips = (trade.entry_price - exit_price) / pip_size

        pip_value = 10.0 * trade.lot_size  # Approximate
        pnl_usd = pnl_pips * pip_value - trade.commission

        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.pnl_pips = round(pnl_pips, 1)
        trade.pnl_usd = round(pnl_usd, 2)
        trade.exit_reason = reason

        self.balance += pnl_usd + trade.commission  # Add back commission since it was already subtracted
        self.balance += pnl_pips * pip_value  # This is the actual P/L without commission
        # Correction: balance = balance + pnl_usd (which already includes commission)
        # Actually let's simplify:
        self.balance = self.balance + pnl_pips * pip_value  # Raw P/L (commission already deducted at entry)

        self.open_trades.remove(trade)
        self.closed_trades.append(trade)

    def _calc_unrealized(self, trade: Trade, current_price: float, pip_size: float) -> float:
        """Calculate unrealized P/L for an open trade."""
        if trade.direction == "buy":
            pips = (current_price - trade.entry_price) / pip_size
        else:
            pips = (trade.entry_price - current_price) / pip_size
        return pips * 10.0 * trade.lot_size

    def _compile_results(self) -> BacktestResult:
        """Compile final backtest results."""
        result = BacktestResult()
        result.trades = self.closed_trades
        result.equity_curve = self.equity_curve
        result.initial_balance = self.initial_balance
        result.final_balance = self.balance
        result.total_trades = len(self.closed_trades)

        if not self.closed_trades:
            return result

        pnls = [t.pnl_usd for t in self.closed_trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        result.winners = len(winners)
        result.losers = len(losers)
        result.win_rate = len(winners) / len(pnls) if pnls else 0
        result.total_pnl = sum(pnls)
        result.avg_trade_pnl = np.mean(pnls)
        result.avg_winner = np.mean(winners) if winners else 0
        result.avg_loser = np.mean(losers) if losers else 0
        result.profit_factor = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf')
        result.max_drawdown_pct = self.max_dd

        # Sharpe ratio (annualized, assuming daily returns)
        if len(pnls) > 1:
            returns = pd.Series(pnls)
            result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # R-multiples
        r_multiples = []
        for t in self.closed_trades:
            risk = abs(t.entry_price - t.sl)
            if risk > 0:
                if t.direction == "buy":
                    r = (t.exit_price - t.entry_price) / risk
                else:
                    r = (t.entry_price - t.exit_price) / risk
                r_multiples.append(r)
        result.avg_r_multiple = np.mean(r_multiples) if r_multiples else 0

        # Consecutive streaks
        max_wins = max_losses = current_wins = current_losses = 0
        for p in pnls:
            if p > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        result.max_consecutive_wins = max_wins
        result.max_consecutive_losses = max_losses

        logger.info(f"Backtest complete: {result.total_trades} trades, "
                    f"WR={result.win_rate:.1%}, PF={result.profit_factor:.2f}, "
                    f"Sharpe={result.sharpe_ratio:.2f}, MaxDD={result.max_drawdown_pct:.1f}%")

        return result

    def print_summary(self, result: BacktestResult):
        """Print a formatted summary of backtest results."""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Initial Balance:  ${result.initial_balance:,.2f}")
        print(f"Final Balance:    ${result.final_balance:,.2f}")
        print(f"Total P/L:        ${result.total_pnl:,.2f}")
        print(f"{'─'*60}")
        print(f"Total Trades:     {result.total_trades}")
        print(f"Winners:          {result.winners} ({result.win_rate:.1%})")
        print(f"Losers:           {result.losers}")
        print(f"Avg Winner:       ${result.avg_winner:,.2f}")
        print(f"Avg Loser:        ${result.avg_loser:,.2f}")
        print(f"Profit Factor:    {result.profit_factor:.2f}")
        print(f"Avg R-Multiple:   {result.avg_r_multiple:.2f}")
        print(f"{'─'*60}")
        print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown:     {result.max_drawdown_pct:.1f}%")
        print(f"Max Consec Wins:  {result.max_consecutive_wins}")
        print(f"Max Consec Losses:{result.max_consecutive_losses}")
        print(f"{'='*60}\n")
