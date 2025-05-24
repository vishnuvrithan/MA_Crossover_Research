"""
research_tools.py: Advanced research tools for evaluating trading strategies
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit


def monte_carlo_simulation(strategy_returns, num_simulations=1000):
    np.random.seed(42)
    results = []
    for _ in range(num_simulations):
        sampled_returns = np.random.choice(strategy_returns.dropna(), size=len(strategy_returns), replace=True)
        cumulative = np.cumprod(1 + sampled_returns)[-1]
        results.append(cumulative)
    return results


def walk_forward_optimization(data: pd.DataFrame, window_size: int, step_size: int, short_range, long_range):
    results = []
    for start in range(0, len(data) - window_size, step_size):
        end = start + window_size
        train = data.iloc[start:end].copy()
        best_sharpe = -np.inf
        best_params = (0, 0)
        for short in short_range:
            for long in long_range:
                if short >= long:
                    continue
                train['SMA_S'] = train['Close'].rolling(short).mean()
                train['SMA_L'] = train['Close'].rolling(long).mean()
                train['Signal'] = np.where(train['SMA_S'] > train['SMA_L'], 1, 0)
                returns = train['Signal'].shift(1) * train['Close'].pct_change()
                sharpe = returns.mean() / returns.std()
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (short, long)
        results.append({'start': start, 'end': end, 'short': best_params[0], 'long': best_params[1], 'sharpe': best_sharpe})
    return pd.DataFrame(results)


def parameter_sensitivity(data: pd.DataFrame, short_range, long_range):
    heatmap_data = np.zeros((len(short_range), len(long_range)))
    for i, short in enumerate(short_range):
        for j, long in enumerate(long_range):
            if short >= long:
                heatmap_data[i][j] = np.nan
                continue
            data['SMA_S'] = data['Close'].rolling(short).mean()
            data['SMA_L'] = data['Close'].rolling(long).mean()
            data['Signal'] = np.where(data['SMA_S'] > data['SMA_L'], 1, 0)
            returns = data['Signal'].shift(1) * data['Close'].pct_change()
            heatmap_data[i][j] = returns.mean() / returns.std()
    return heatmap_data
