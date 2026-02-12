"""
Walk-Forward Optimization (WFO)
Train on N months, test on M months, step forward, repeat
Uses Bayesian optimization for parameter search
"""
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from copy import deepcopy
from .engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)

# Try to import scikit-optimize for Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logger.warning("scikit-optimize not available - falling back to random search")


class WalkForwardOptimizer:
    def __init__(self, strategy_class, param_space: dict,
                 train_months: int = 6, test_months: int = 3,
                 step_months: int = 3,
                 initial_balance: float = 10000.0,
                 optimization_metric: str = "sharpe_ratio"):
        """
        strategy_class: class that accepts config dict in __init__
        param_space: dict of param_name -> (min, max, type)
            e.g. {"ema_fast": (10, 30, "int"), "atr_mult": (1.0, 2.5, "float")}
        """
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.initial_balance = initial_balance
        self.optimization_metric = optimization_metric
        self.results = []

    def run(self, bars_dict: dict, symbol: str = "EURUSD",
            n_calls: int = 30, digits: int = 5) -> dict:
        """
        Run walk-forward optimization.
        bars_dict: multi-timeframe data dict
        Returns summary with in-sample and out-of-sample results.
        """
        # Get date range from primary timeframe
        primary_df = None
        for tf in ["M5", "M15", "H1"]:
            if tf in bars_dict:
                primary_df = bars_dict[tf]
                break

        if primary_df is None:
            raise ValueError("No valid data in bars_dict")

        start_date = primary_df.index[0]
        end_date = primary_df.index[-1]

        train_delta = pd.DateOffset(months=self.train_months)
        test_delta = pd.DateOffset(months=self.test_months)
        step_delta = pd.DateOffset(months=self.step_months)

        current_start = start_date
        window_results = []

        logger.info(f"Starting WFO: {start_date.date()} to {end_date.date()}")
        logger.info(f"Train: {self.train_months}m, Test: {self.test_months}m, Step: {self.step_months}m")

        window_num = 0
        while current_start + train_delta + test_delta <= end_date:
            window_num += 1
            train_end = current_start + train_delta
            test_end = train_end + test_delta

            logger.info(f"\nWindow {window_num}: Train [{current_start.date()} - {train_end.date()}], "
                        f"Test [{train_end.date()} - {test_end.date()}]")

            # Split data
            train_data = {tf: df[(df.index >= current_start) & (df.index < train_end)]
                         for tf, df in bars_dict.items()}
            test_data = {tf: df[(df.index >= train_end) & (df.index < test_end)]
                        for tf, df in bars_dict.items()}

            # Optimize on train data
            best_params = self._optimize(train_data, symbol, n_calls, digits)

            # Test with best params on out-of-sample data
            strategy = self.strategy_class(best_params)
            engine = BacktestEngine(initial_balance=self.initial_balance)
            oos_result = engine.run(strategy, test_data, symbol, digits)

            # Also get in-sample result for comparison
            engine_is = BacktestEngine(initial_balance=self.initial_balance)
            is_result = engine_is.run(strategy, train_data, symbol, digits)

            window_results.append({
                "window": window_num,
                "train_start": str(current_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(train_end.date()),
                "test_end": str(test_end.date()),
                "best_params": best_params,
                "is_sharpe": is_result.sharpe_ratio,
                "is_pf": is_result.profit_factor,
                "is_wr": is_result.win_rate,
                "is_trades": is_result.total_trades,
                "oos_sharpe": oos_result.sharpe_ratio,
                "oos_pf": oos_result.profit_factor,
                "oos_wr": oos_result.win_rate,
                "oos_trades": oos_result.total_trades,
                "oos_pnl": oos_result.total_pnl,
                "oos_max_dd": oos_result.max_drawdown_pct,
            })

            logger.info(f"  IS Sharpe={is_result.sharpe_ratio:.2f}, OOS Sharpe={oos_result.sharpe_ratio:.2f}")

            current_start += step_delta

        self.results = window_results
        summary = self._compile_summary(window_results)
        return summary

    def _optimize(self, train_data: dict, symbol: str, n_calls: int, digits: int) -> dict:
        """Optimize parameters on training data."""
        if SKOPT_AVAILABLE:
            return self._bayesian_optimize(train_data, symbol, n_calls, digits)
        return self._random_search(train_data, symbol, n_calls, digits)

    def _bayesian_optimize(self, train_data, symbol, n_calls, digits) -> dict:
        """Use Bayesian optimization to find best parameters."""
        space = []
        param_names = []
        for name, (low, high, ptype) in self.param_space.items():
            param_names.append(name)
            if ptype == "int":
                space.append(Integer(int(low), int(high), name=name))
            else:
                space.append(Real(float(low), float(high), name=name))

        def objective(params):
            config = dict(zip(param_names, params))
            strategy = self.strategy_class(config)
            engine = BacktestEngine(initial_balance=self.initial_balance)
            result = engine.run(strategy, train_data, symbol, digits)

            metric = getattr(result, self.optimization_metric, 0)
            return -metric  # Minimize negative = maximize

        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42, verbose=False)

        best_params = dict(zip(param_names, result.x))
        logger.info(f"  Best params: {best_params}")
        return best_params

    def _random_search(self, train_data, symbol, n_calls, digits) -> dict:
        """Fallback random search when skopt not available."""
        best_score = -np.inf
        best_params = {}

        for i in range(n_calls):
            config = {}
            for name, (low, high, ptype) in self.param_space.items():
                if ptype == "int":
                    config[name] = np.random.randint(int(low), int(high) + 1)
                else:
                    config[name] = np.random.uniform(float(low), float(high))

            strategy = self.strategy_class(config)
            engine = BacktestEngine(initial_balance=self.initial_balance)
            result = engine.run(strategy, train_data, symbol, digits)

            metric = getattr(result, self.optimization_metric, 0)
            if metric > best_score:
                best_score = metric
                best_params = config

        logger.info(f"  Best params (random): {best_params}")
        return best_params

    def _compile_summary(self, window_results: list) -> dict:
        """Compile overall WFO summary."""
        if not window_results:
            return {"error": "No windows completed"}

        oos_sharpes = [w["oos_sharpe"] for w in window_results]
        oos_pfs = [w["oos_pf"] for w in window_results]
        oos_wrs = [w["oos_wr"] for w in window_results]
        is_sharpes = [w["is_sharpe"] for w in window_results]

        # Overfitting detection
        avg_is_sharpe = np.mean(is_sharpes)
        avg_oos_sharpe = np.mean(oos_sharpes)
        degradation = (avg_is_sharpe - avg_oos_sharpe) / avg_is_sharpe * 100 if avg_is_sharpe > 0 else 0

        summary = {
            "total_windows": len(window_results),
            "avg_oos_sharpe": round(float(np.mean(oos_sharpes)), 3),
            "avg_oos_pf": round(float(np.mean(oos_pfs)), 3),
            "avg_oos_wr": round(float(np.mean(oos_wrs)), 3),
            "avg_is_sharpe": round(float(avg_is_sharpe), 3),
            "sharpe_degradation_pct": round(float(degradation), 1),
            "overfitting_detected": degradation > 50,
            "profitable_windows": sum(1 for w in window_results if w["oos_pnl"] > 0),
            "total_oos_pnl": round(sum(w["oos_pnl"] for w in window_results), 2),
            "windows": window_results,
        }

        logger.info(f"\nWFO Summary: OOS Sharpe={summary['avg_oos_sharpe']}, "
                    f"Degradation={summary['sharpe_degradation_pct']}%, "
                    f"Overfitting={'YES' if summary['overfitting_detected'] else 'NO'}")

        return summary

    def save_results(self, filepath: str):
        """Save WFO results to JSON."""
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
