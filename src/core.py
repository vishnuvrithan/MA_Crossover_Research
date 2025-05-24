"""
core.py: Moving Average crossover logic + basic backtest
"""

import pandas as pd
import numpy as np

class StrategyCore:
    def calculate_signals(self, data: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
        data = data.copy()
        data['SMA_Short'] = data['Close'].rolling(short_window).mean()
        data['SMA_Long'] = data['Close'].rolling(long_window).mean()
        data['Signal'] = np.where(data['SMA_Short'] > data['SMA_Long'], 1, 0)
        data['Position'] = data['Signal'].diff()
        return data

    def backtest(self, data: pd.DataFrame, initial_capital=100000, transaction_cost=0.0005) -> pd.DataFrame:
        data = data.copy()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']
        trades = data['Position'].abs().sum()
        cost_per_trade = (trades * transaction_cost) / trades if trades > 0 else 0
        data['Strategy_Return'] = np.where(data['Position'] != 0, data['Strategy_Return'] - cost_per_trade, data['Strategy_Return'])
        data['Cumulative_Market'] = (1 + data['Daily_Return']).cumprod()
        data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
        data['Portfolio_Value'] = initial_capital * data['Cumulative_Strategy']
        return data