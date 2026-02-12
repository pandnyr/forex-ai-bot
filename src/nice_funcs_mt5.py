"""
MT5 Exchange Module - Core Trading Functions
Unified interface matching MoonDev's pattern, adapted for Forex/MT5
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import logging

logger = logging.getLogger(__name__)

# MT5 Timeframe mapping
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}


def initialize(path: str = None, login: int = None, password: str = None, server: str = None) -> bool:
    """Initialize MT5 connection."""
    kwargs = {}
    if path:
        kwargs["path"] = path
    if not mt5.initialize(**kwargs):
        logger.error(f"MT5 initialization failed: {mt5.last_error()}")
        return False

    if login and password and server:
        if not mt5.login(login=login, password=password, server=server):
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            return False

    info = mt5.account_info()
    if info:
        logger.info(f"Connected: {info.server}, Balance: ${info.balance:.2f}, Leverage: 1:{info.leverage}")
    return True


def shutdown():
    """Shutdown MT5 connection."""
    mt5.shutdown()
    logger.info("MT5 connection closed")


def get_account_balance() -> float:
    """Get current account balance."""
    info = mt5.account_info()
    return info.balance if info else 0.0


def get_account_equity() -> float:
    """Get current account equity."""
    info = mt5.account_info()
    return info.equity if info else 0.0


def get_account_info() -> dict:
    """Get full account information."""
    info = mt5.account_info()
    if not info:
        return {}
    return {
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "free_margin": info.margin_free,
        "margin_level": info.margin_level,
        "profit": info.profit,
        "leverage": info.leverage,
        "currency": info.currency,
        "server": info.server,
    }


def get_spread(symbol: str) -> float:
    """Get current spread in pips."""
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return 999.0
    info = mt5.symbol_info(symbol)
    if not info:
        return 999.0
    point = info.point
    digits = info.digits
    spread_points = tick.ask - tick.bid
    # Convert to pips (for 5-digit brokers, 1 pip = 10 points)
    if digits == 5 or digits == 3:
        return spread_points / (point * 10)
    return spread_points / point


def get_pip_value(symbol: str, lot_size: float = 0.01) -> float:
    """Get pip value in account currency for given lot size."""
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.0
    point = info.point
    digits = info.digits
    pip_size = point * 10 if (digits == 5 or digits == 3) else point
    # Calculate using MT5's built-in function
    profit = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, symbol,
                                    lot_size, info.bid, info.bid + pip_size)
    return abs(profit) if profit else 0.0


def calculate_lot_size(symbol: str, risk_usd: float, sl_pips: float) -> float:
    """Calculate lot size based on risk amount and SL distance in pips."""
    if sl_pips <= 0:
        return 0.01  # Minimum lot

    pip_value = get_pip_value(symbol, 1.0)  # Pip value for 1 standard lot
    if pip_value <= 0:
        return 0.01

    lot = risk_usd / (sl_pips * pip_value)

    # Clamp to broker limits
    info = mt5.symbol_info(symbol)
    if info:
        lot = max(info.volume_min, min(lot, info.volume_max))
        # Round to volume step
        step = info.volume_step
        lot = round(lot / step) * step
        lot = round(lot, 2)

    return max(0.01, lot)


def get_bars(symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
    """Get OHLCV bars as DataFrame."""
    tf = TIMEFRAME_MAP.get(timeframe)
    if tf is None:
        logger.error(f"Invalid timeframe: {timeframe}")
        return pd.DataFrame()

    # Ensure symbol is selected
    mt5.symbol_select(symbol, True)

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        logger.error(f"No data for {symbol} {timeframe}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def get_tick(symbol: str) -> dict:
    """Get latest tick data."""
    mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return {}
    return {
        "bid": tick.bid,
        "ask": tick.ask,
        "last": tick.last,
        "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
    }


def place_order(symbol: str, order_type: str, lot: float,
                sl: float = 0.0, tp: float = 0.0,
                comment: str = "forex_bot", magic: int = 234000) -> dict:
    """
    Place a market order.
    order_type: 'buy' or 'sell'
    Returns dict with ticket, price, etc.
    """
    mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return {"success": False, "error": f"No tick data for {symbol}"}

    info = mt5.symbol_info(symbol)
    if not info:
        return {"success": False, "error": f"No symbol info for {symbol}"}

    if order_type.lower() == "buy":
        mt5_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    elif order_type.lower() == "sell":
        mt5_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        return {"success": False, "error": f"Invalid order type: {order_type}"}

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        return {"success": False, "error": str(mt5.last_error())}

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {
            "success": False,
            "retcode": result.retcode,
            "error": f"Order failed: retcode={result.retcode}, comment={result.comment}",
        }

    logger.info(f"Order placed: {order_type.upper()} {lot} {symbol} @ {result.price}, "
                f"SL={sl}, TP={tp}, Ticket={result.order}")
    return {
        "success": True,
        "ticket": result.order,
        "price": result.price,
        "volume": result.volume,
        "symbol": symbol,
        "type": order_type,
    }


def market_buy(symbol: str, usd_amount: float, sl_pips: float = 15.0,
               tp_pips: float = 30.0, comment: str = "forex_bot") -> dict:
    """Market buy with automatic lot sizing based on USD risk amount."""
    lot = calculate_lot_size(symbol, usd_amount, sl_pips)
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if not tick or not info:
        return {"success": False, "error": "Cannot get symbol data"}

    point = info.point
    digits = info.digits
    pip_size = point * 10 if (digits == 5 or digits == 3) else point

    sl_price = round(tick.ask - sl_pips * pip_size, digits)
    tp_price = round(tick.ask + tp_pips * pip_size, digits)

    return place_order(symbol, "buy", lot, sl=sl_price, tp=tp_price, comment=comment)


def market_sell(symbol: str, usd_amount: float, sl_pips: float = 15.0,
                tp_pips: float = 30.0, comment: str = "forex_bot") -> dict:
    """Market sell with automatic lot sizing based on USD risk amount."""
    lot = calculate_lot_size(symbol, usd_amount, sl_pips)
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if not tick or not info:
        return {"success": False, "error": "Cannot get symbol data"}

    point = info.point
    digits = info.digits
    pip_size = point * 10 if (digits == 5 or digits == 3) else point

    sl_price = round(tick.bid + sl_pips * pip_size, digits)
    tp_price = round(tick.bid - tp_pips * pip_size, digits)

    return place_order(symbol, "sell", lot, sl=sl_price, tp=tp_price, comment=comment)


def modify_order(ticket: int, sl: float = None, tp: float = None) -> dict:
    """Modify SL/TP of an existing position."""
    position = mt5.positions_get(ticket=ticket)
    if not position or len(position) == 0:
        return {"success": False, "error": f"Position {ticket} not found"}

    pos = position[0]
    new_sl = sl if sl is not None else pos.sl
    new_tp = tp if tp is not None else pos.tp

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": ticket,
        "sl": new_sl,
        "tp": new_tp,
    }

    result = mt5.order_send(request)
    if result is None:
        return {"success": False, "error": str(mt5.last_error())}

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "retcode": result.retcode, "error": result.comment}

    logger.info(f"Modified ticket {ticket}: SL={new_sl}, TP={new_tp}")
    return {"success": True, "ticket": ticket, "sl": new_sl, "tp": new_tp}


def trail_stop(ticket: int, trail_pips: float) -> dict:
    """Update trailing stop on a position."""
    position = mt5.positions_get(ticket=ticket)
    if not position or len(position) == 0:
        return {"success": False, "error": f"Position {ticket} not found"}

    pos = position[0]
    info = mt5.symbol_info(pos.symbol)
    tick = mt5.symbol_info_tick(pos.symbol)
    if not info or not tick:
        return {"success": False, "error": "Cannot get symbol data"}

    point = info.point
    digits = info.digits
    pip_size = point * 10 if (digits == 5 or digits == 3) else point
    trail_distance = trail_pips * pip_size

    if pos.type == mt5.ORDER_TYPE_BUY:
        new_sl = round(tick.bid - trail_distance, digits)
        if new_sl > pos.sl and new_sl < tick.bid:
            return modify_order(ticket, sl=new_sl)
    else:
        new_sl = round(tick.ask + trail_distance, digits)
        if pos.sl == 0 or (new_sl < pos.sl and new_sl > tick.ask):
            return modify_order(ticket, sl=new_sl)

    return {"success": False, "error": "Trail stop not moved (price not favorable enough)"}


def get_position(symbol: str = None) -> list:
    """Get open positions. If symbol specified, filter by symbol."""
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()

    if positions is None or len(positions) == 0:
        return []

    result = []
    for pos in positions:
        result.append({
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "type": "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell",
            "volume": pos.volume,
            "price_open": pos.price_open,
            "price_current": pos.price_current,
            "sl": pos.sl,
            "tp": pos.tp,
            "profit": pos.profit,
            "swap": pos.swap,
            "magic": pos.magic,
            "comment": pos.comment,
            "time": datetime.fromtimestamp(pos.time, tz=timezone.utc),
        })
    return result


def get_bot_positions(magic: int = 234000) -> list:
    """Get only positions opened by our bot (filtered by magic number)."""
    all_positions = get_position()
    return [p for p in all_positions if p["magic"] == magic]


def close_position(ticket: int) -> dict:
    """Close a specific position by ticket."""
    position = mt5.positions_get(ticket=ticket)
    if not position or len(position) == 0:
        return {"success": False, "error": f"Position {ticket} not found"}

    pos = position[0]
    tick = mt5.symbol_info_tick(pos.symbol)
    if not tick:
        return {"success": False, "error": f"No tick data for {pos.symbol}"}

    # Reverse the trade
    if pos.type == mt5.ORDER_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": 10,
        "magic": pos.magic,
        "comment": "close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        return {"success": False, "error": str(mt5.last_error())}

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "retcode": result.retcode, "error": result.comment}

    logger.info(f"Closed position {ticket}: {pos.volume} {pos.symbol} @ {result.price}")
    return {"success": True, "ticket": ticket, "price": result.price, "profit": pos.profit}


def chunk_kill(symbol: str, magic: int = 234000) -> list:
    """Close all positions for a symbol (matching magic number)."""
    positions = get_bot_positions(magic)
    results = []
    for pos in positions:
        if pos["symbol"] == symbol:
            result = close_position(pos["ticket"])
            results.append(result)
            time.sleep(0.5)
    return results


def close_all(magic: int = 234000) -> list:
    """Close ALL bot positions."""
    positions = get_bot_positions(magic)
    results = []
    for pos in positions:
        result = close_position(pos["ticket"])
        results.append(result)
        time.sleep(0.5)
    return results


def get_trade_history(days: int = 30) -> pd.DataFrame:
    """Get trade history for the last N days."""
    from_date = datetime.now(timezone.utc) - pd.Timedelta(days=days)
    to_date = datetime.now(timezone.utc)

    deals = mt5.history_deals_get(from_date, to_date)
    if deals is None or len(deals) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def get_daily_pnl(magic: int = 234000) -> float:
    """Calculate today's P/L for bot positions."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + pd.Timedelta(days=1)

    deals = mt5.history_deals_get(today, tomorrow)
    if deals is None:
        return 0.0

    pnl = sum(d.profit + d.swap + d.commission for d in deals if d.magic == magic)
    # Add unrealized P/L from open positions
    positions = get_bot_positions(magic)
    pnl += sum(p["profit"] + p["swap"] for p in positions)
    return pnl


def get_daily_trade_count(magic: int = 234000) -> int:
    """Count trades opened today."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + pd.Timedelta(days=1)

    deals = mt5.history_deals_get(today, tomorrow)
    if deals is None:
        return 0

    # Count entry deals only (not exits)
    return sum(1 for d in deals if d.magic == magic and d.entry == mt5.DEAL_ENTRY_IN)


def set_leverage(symbol: str, leverage: int) -> bool:
    """Note: Leverage is set at account level in MT5, not per-symbol.
    This is a no-op placeholder. Configure leverage through your broker."""
    logger.warning(f"Leverage must be set through broker account settings. "
                   f"Requested: 1:{leverage} for {symbol}")
    return True


def get_symbol_info(symbol: str) -> dict:
    """Get full symbol information."""
    mt5.symbol_select(symbol, True)
    info = mt5.symbol_info(symbol)
    if not info:
        return {}
    return {
        "symbol": info.name,
        "digits": info.digits,
        "point": info.point,
        "spread": info.spread,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
        "trade_contract_size": info.trade_contract_size,
        "swap_long": info.swap_long,
        "swap_short": info.swap_short,
    }
