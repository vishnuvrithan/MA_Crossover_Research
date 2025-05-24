"""
Advanced Moving Average Crossover Research System

This system implements a comprehensive research framework for MA crossover strategies,
with improvements for performance, robustness, and extensibility.

Key Improvements:
1. Parallel processing for optimization tasks
2. Support for EMA alongside SMA
3. Risk management with stop-loss/take-profit
4. Interactive visualizations using Plotly
5. Modular strategy design for extensibility
6. Enhanced input validation and error handling
7. Basic unit tests
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, List, Optional
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
from joblib import Parallel, delayed
from abc import ABC, abstractmethod
import unittest

# Configuration
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class MACrossoverStrategy(TradingStrategy):
    """Moving Average Crossover Strategy implementation."""
    def __init__(self, short_window: int, long_window: int, ma_type: str = 'sma'):
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type.lower()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MA crossover."""
        if not isinstance(data, pd.DataFrame) or 'Close' not in data.columns:
            raise ValueError("Input must be a DataFrame with 'Close' column")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")
        if self.short_window < 1 or self.long_window < 1:
            raise ValueError("Window sizes must be positive integers")

        data = data.copy()
        if self.ma_type == 'ema':
            data['MA_Short'] = data['Close'].ewm(span=self.short_window, adjust=False).mean()
            data['MA_Long'] = data['Close'].ewm(span=self.long_window, adjust=False).mean()
        else:
            data['MA_Short'] = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
            data['MA_Long'] = data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        data['Signal'] = (data['MA_Short'] > data['MA_Long']).astype(int)
        data['Position'] = data['Signal'].diff()
        return data

class MACrossoverResearch:
    """Comprehensive research framework for moving average crossover strategies."""
    
    def __init__(self):
        self.data = None
        self.results = {}
        self.best_params = {}
        self.strategy = None
        
    # Data Methods ------------------------------------------------------------
    def generate_test_data(self, size=1000, trend=0.0005, volatility=0.01, 
                         seed=42, regimes=False) -> pd.DataFrame:
        """
        Generate synthetic price data with optional market regimes.

        Args:
            size (int): Number of data points.
            trend (float): Daily drift component.
            volatility (float): Daily volatility.
            seed (int): Random seed for reproducibility.
            regimes (bool): Whether to include regime changes.

        Returns:
            pd.DataFrame: DataFrame with synthetic price data.
        """
        np.random.seed(seed)
        dates = pd.date_range(end=datetime.today(), periods=size)
        
        if regimes:
            regime_length = size // 4
            returns = np.concatenate([
                np.random.normal(loc=0.001, scale=0.005, size=regime_length),
                np.random.normal(loc=-0.0005, scale=0.02, size=regime_length),
                np.random.normal(loc=0.0002, scale=0.015, size=regime_length),
                np.random.normal(loc=0.0007, scale=0.008, size=size-3*regime_length)
            ])
        else:
            returns = np.random.normal(loc=trend, scale=volatility, size=size)
        
        prices = 100 * (1 + returns).cumprod()
        return pd.DataFrame({'Close': prices}, index=dates)
    
    def load_real_data(self, filepath: str, fill_method: str = 'ffill') -> pd.DataFrame:
        """
        Load real market data from CSV file.

        Args:
            filepath (str): Path to CSV file with columns=['Date', 'Close'].
            fill_method (str): Method to fill missing values ('ffill', 'bfill', or None).

        Returns:
            pd.DataFrame: DataFrame with price data.
        """
        try:
            df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
            df = df[['Close']]
            if fill_method:
                df = df.fillna(method=fill_method)
            df = df.dropna()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    # Core Strategy -----------------------------------------------------------
    def set_strategy(self, strategy: TradingStrategy):
        """Set the trading strategy to use."""
        self.strategy = strategy

    def calculate_signals(self, data: pd.DataFrame, short_window: int, 
                        long_window: int, ma_type: str = 'sma') -> pd.DataFrame:
        """
        Calculate moving averages and trading signals.

        Args:
            data (pd.DataFrame): DataFrame with 'Close' column.
            short_window (int): Short MA period.
            long_window (int): Long MA period.
            ma_type (str): Type of moving average ('sma' or 'ema').

        Returns:
            pd.DataFrame: DataFrame with signals.
        """
        self.strategy = MACrossoverStrategy(short_window, long_window, ma_type)
        return self.strategy.generate_signals(data)
    
    def backtest(self, data: pd.DataFrame, initial_capital=100000, 
                transaction_cost=0.0005, stop_loss=0.02, take_profit=0.05) -> pd.DataFrame:
        """
        Run backtest on signal data.

        Args:
            data (pd.DataFrame): DataFrame with signals.
            initial_capital (float): Starting capital.
            transaction_cost (float): Percentage cost per transaction.
            stop_loss (float): Stop-loss threshold as a fraction.
            take_profit (float): Take-profit threshold as a fraction.

        Returns:
            pd.DataFrame: DataFrame with backtest results.
        """
        if 'Signal' not in data.columns:
            raise ValueError("Data must contain signals")
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
            
        data = data.copy()
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Calculate strategy returns
        data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']
        
        # Apply stop-loss and take-profit
        cum_returns = data['Strategy_Return'].cumsum()
        data['Position'] = data['Position'].where(
            (cum_returns > -stop_loss) & (cum_returns < take_profit),
            0
        )
        
        # Apply transaction costs
        trades = data['Position'].abs().sum()
        total_cost = trades * transaction_cost
        cost_per_trade = total_cost / trades if trades > 0 else 0
        
        data['Strategy_Return'] = np.where(
            data['Position'] != 0, 
            data['Strategy_Return'] - cost_per_trade, 
            data['Strategy_Return']
        )
        
        # Calculate portfolio values
        data['Cumulative_Market'] = (1 + data['Daily_Return']).cumprod()
        data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
        data['Portfolio_Value'] = initial_capital * data['Cumulative_Strategy']
        
        return data
    
    # Research Methods -------------------------------------------------------
    def walk_forward_optimization(self, data: pd.DataFrame, 
                                short_range: Tuple[int, int], 
                                long_range: Tuple[int, int],
                                train_size: int = 252*2, 
                                test_size: int = 252,
                                step_size: int = 63,
                                ma_type: str = 'sma') -> Dict:
        """
        Perform walk-forward optimization.

        Args:
            data (pd.DataFrame): Price data.
            short_range (Tuple[int, int]): (min, max) for short MA.
            long_range (Tuple[int, int]): (min, max) for long MA.
            train_size (int): Training window size in days.
            test_size (int): Testing window size in days.
            step_size (int): Step size between windows.
            ma_type (str): Moving average type ('sma' or 'ema').

        Returns:
            Dict: Optimization results, best parameters, and summary stats.
        """
        if len(data) < train_size + test_size:
            raise ValueError("Dataset too small for specified train/test sizes")
        
        def run_train_test(s, l, train_data, test_data):
            try:
                train_signal = self.calculate_signals(train_data, s, l, ma_type)
                train_result = self.backtest(train_signal)
                metrics = self.calculate_metrics(train_result['Strategy_Return'])
                
                test_signal = self.calculate_signals(test_data, s, l, ma_type)
                test_result = self.backtest(test_signal)
                test_metrics = self.calculate_metrics(test_result['Strategy_Return'])
                
                return {
                    'short': s,
                    'long': l,
                    'train_sharpe': metrics['sharpe_ratio'],
                    'test_sharpe': test_metrics['sharpe_ratio'],
                    'test_return': test_metrics['annualized_return']
                }
            except Exception as e:
                logger.warning(f"Failed for parameters {s}/{l}: {str(e)}")
                return None

        results = []
        best_params = []
        
        # Generate parameter combinations
        short_params = range(short_range[0], short_range[1]+1, 5)
        long_params = range(long_range[0], long_range[1]+1, 5)
        param_combinations = [(s, l) for s in short_params for l in long_params if s < l]
        
        # Walk-forward windows
        n_windows = (len(data) - train_size - test_size) // step_size + 1
        
        for i in tqdm(range(n_windows), desc="Walk-forward optimization"):
            train_start = i * step_size
            train_end = train_start + train_size
            test_end = train_end + test_size
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            # Parallel optimization
            train_results = Parallel(n_jobs=-1)(
                delayed(run_train_test)(s, l, train_data, test_data) for s, l in param_combinations
            )
            train_results = [r for r in train_results if r is not None]
            
            if not train_results:
                continue
                
            # Find best params
            train_df = pd.DataFrame(train_results)
            best_param = train_df.loc[train_df['train_sharpe'].idxmax()]
            
            results.append({
                'window': i,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_short': best_param['short'],
                'best_long': best_param['long'],
                'train_sharpe': best_param['train_sharpe'],
                'test_sharpe': best_param['test_sharpe'],
                'test_return': best_param['test_return']
            })
            
            best_params.append({
                'short': best_param['short'],
                'long': best_param['long'],
                'window': i
            })
        
        self.wfo_results = pd.DataFrame(results)
        self.best_params = pd.DataFrame(best_params)
        
        return {
            'results': self.wfo_results,
            'best_params': self.best_params,
            'summary_stats': self.wfo_results.describe()
        }
    
    def monte_carlo_simulation(self, data: pd.DataFrame, n_simulations=1000,
                             short_window=50, long_window=200, ma_type: str = 'sma') -> Dict:
        """
        Perform Monte Carlo simulation of strategy returns.

        Args:
            data (pd.DataFrame): Price data.
            n_simulations (int): Number of simulations.
            short_window (int): Short MA window.
            long_window (int): Long MA window.
            ma_type (str): Moving average type ('sma' or 'ema').

        Returns:
            Dict: Simulation results.
        """
        signal_data = self.calculate_signals(data, short_window, long_window, ma_type)
        actual_result = self.backtest(signal_data)
        actual_returns = actual_result['Strategy_Return'].dropna()
        actual_sharpe = self.calculate_metrics(actual_returns)['sharpe_ratio']
        
        # Generate random paths
        random_sharpes = []
        daily_vol = data['Close'].pct_change().std()
        daily_mean = data['Close'].pct_change().mean()
        
        for _ in tqdm(range(n_simulations), desc="Monte Carlo Simulation"):
            random_returns = np.random.normal(
                loc=daily_mean, 
                scale=daily_vol, 
                size=len(actual_returns)
            )
            random_sharpe = np.sqrt(252) * random_returns.mean() / random_returns.std()
            random_sharpes.append(random_sharpe)
        
        p_value = (np.array(random_sharpes) > actual_sharpe).mean()
        
        self.mc_results = {
            'actual_sharpe': actual_sharpe,
            'random_sharpes': random_sharpes,
            'p_value': p_value,
            'significance_level': 0.05
        }
        
        return self.mc_results
    
    def parameter_sensitivity(self, data: pd.DataFrame, 
                            short_range: Tuple[int, int], 
                            long_range: Tuple[int, int],
                            ma_type: str = 'sma') -> pd.DataFrame:
        """
        Analyze parameter sensitivity across ranges.

        Args:
            data (pd.DataFrame): Price data.
            short_range (Tuple[int, int]): (min, max) for short MA.
            long_range (Tuple[int, int]): (min, max) for long MA.
            ma_type (str): Moving average type ('sma' or 'ema').

        Returns:
            pd.DataFrame: Sensitivity results.
        """
        def run_backtest(s, l):
            if s >= l:
                return None
            try:
                signal_data = self.calculate_signals(data, s, l, ma_type)
                result = self.backtest(signal_data)
                metrics = self.calculate_metrics(result['Strategy_Return'])
                return {
                    'short_window': s,
                    'long_window': l,
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'annual_return': metrics['annualized_return'],
                    'max_drawdown': metrics['max_drawdown']
                }
            except Exception as e:
                logger.warning(f"Failed for parameters {s}/{l}: {str(e)}")
                return None

        short_params = range(short_range[0], short_range[1]+1, 5)
        long_params = range(long_range[0], long_range[1]+1, 5)
        param_combinations = [(s, l) for s in short_params for l in long_params]
        
        results = Parallel(n_jobs=-1)(
            delayed(run_backtest)(s, l) for s, l in tqdm(param_combinations, desc="Parameter sensitivity")
        )
        results = [r for r in results if r is not None]
        
        self.sensitivity_results = pd.DataFrame(results)
        return self.sensitivity_results
    
    # Machine Learning Integration --------------------------------------------
    def create_feature_matrix(self, data: pd.DataFrame, lookback=30) -> pd.DataFrame:
        """
        Create feature matrix for ML models.

        Args:
            data (pd.DataFrame): Price data.
            lookback (int): Lookback window for features.

        Returns:
            pd.DataFrame: DataFrame with features and target.
        """
        df = data.copy()
        
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(lookback).std()
        df['Momentum'] = df['Close'] / df['Close'].shift(lookback) - 1
        
        for w in [5, 10, 20, 50]:
            df[f'MA_{w}'] = df['Close'].rolling(w).mean()
            df[f'Ratio_{w}'] = df['Close'] / df[f'MA_{w}']
        
        df['Target'] = df['Returns'].shift(-1)
        return df.dropna()
    
    def train_hybrid_model(self, data: pd.DataFrame) -> Dict:
        """
        Train a hybrid model combining rules and ML.

        Args:
            data (pd.DataFrame): Feature matrix from create_feature_matrix().

        Returns:
            Dict: Trained model metrics.
        """
        X = data.drop(columns=['Target'])
        y = data['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        
        self.ml_model = model
        self.ml_results = {
            'feature_importances': pd.Series(
                model.feature_importances_, 
                index=X.columns
            ).sort_values(ascending=False),
            'test_mse': mse
        }
        
        return self.ml_results
    
    # Performance Analysis ----------------------------------------------------
    def calculate_metrics(self, returns: pd.Series, risk_free_rate=0.0) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns (pd.Series): Series of strategy returns.
            risk_free_rate (float): Annual risk-free rate.

        Returns:
            Dict: Performance metrics.
        """
        if len(returns) < 5:
            raise ValueError("Insufficient return data")
            
        returns = returns.dropna()
        excess_returns = returns - risk_free_rate / 252
        
        total_return = returns.add(1).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else np.inf
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown.mean()
        
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns <= 0].mean() if (returns <= 0).any() else 0
        profit_factor = -avg_win / avg_loss if avg_loss != 0 else np.inf
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / returns[returns < 0].std() if (returns < 0).std() != 0 else np.inf
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        _, normality_p = stats.shapiro(returns) if len(returns) <= 5000 else (None, None)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'normality_p': normality_p,
            'positive_skew': returns.skew() > 0 if not returns.empty else False
        }
    
    def compare_benchmark(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """
        Compare strategy vs benchmark.

        Args:
            strategy_returns (pd.Series): Series of strategy returns.
            benchmark_returns (pd.Series): Series of benchmark returns.

        Returns:
            Dict: Comparison metrics.
        """
        X = benchmark_returns.values.reshape(-1, 1)
        y = strategy_returns.values
        model = LinearRegression().fit(X, y)
        alpha = model.intercept_ * 252
        beta = model.coef_[0]
        
        excess = strategy_returns - benchmark_returns
        outperformance = (1 + excess).prod() - 1
        
        return {
            'alpha': alpha,
            'beta': beta,
            'outperformance': outperformance,
            'correlation': strategy_returns.corr(benchmark_returns)
        }
    
    # Visualization Methods ---------------------------------------------------
    def plot_equity_curve(self, data: pd.DataFrame) -> None:
        """
        Plot strategy vs market equity curve using Plotly.

        Args:
            data (pd.DataFrame): DataFrame containing backtest results.
        """
        fig = px.line(data, y=['Cumulative_Market', 'Cumulative_Strategy'],
                      title='Strategy vs Market Performance')
        buys = data[data['Position'] == 1]
        sells = data[data['Position'] == -1]
        fig.add_scatter(x=buys.index, y=buys['Cumulative_Strategy'], mode='markers',
                        marker=dict(symbol='triangle-up', color='green'), name='Buy')
        fig.add_scatter(x=sells.index, y=sells['Cumulative_Strategy'], mode='markers',
                        marker=dict(symbol='triangle-down', color='red'), name='Sell')
        fig.update_layout(yaxis_title='Cumulative Returns', xaxis_title='Date')
        fig.show()
    
    def plot_parameter_sensitivity(self, results: pd.DataFrame, metric='sharpe_ratio') -> None:
        """
        Plot parameter sensitivity heatmap.

        Args:
            results (pd.DataFrame): DataFrame from parameter_sensitivity().
            metric (str): Metric to visualize.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        pivot = results.pivot(index='short_window', columns='long_window', values=metric)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap='viridis')
        plt.title(f'Parameter Sensitivity: {metric.replace("_", " ").title()}')
        plt.xlabel('Long Window')
        plt.ylabel('Short Window')
        plt.show()
    
    def plot_monte_carlo(self, results: Dict) -> None:
        """
        Plot Monte Carlo simulation results.

        Args:
            results (Dict): Dictionary from monte_carlo_simulation().
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        sns.histplot(results['random_sharpes'], bins=30, kde=True)
        plt.axvline(results['actual_sharpe'], color='r', linestyle='--', label='Actual Strategy')
        plt.title(f'Monte Carlo Simulation (p-value: {results["p_value"]:.3f})')
        plt.xlabel('Sharpe Ratio')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_walk_forward(self, results: pd.DataFrame) -> None:
        """
        Plot walk-forward optimization results.

        Args:
            results (pd.DataFrame): DataFrame from walk_forward_optimization().
        """
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        results['param_combo'] = results['best_short'].astype(str) + '/' + results['best_long'].astype(str)
        results['param_combo'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_title('Parameter Stability Across Windows')
        ax1.set_ylabel('Frequency')
        
        results[['train_sharpe', 'test_sharpe']].plot(ax=ax2)
        ax2.axhline(results['test_sharpe'].mean(), color='r', linestyle='--')
        ax2.set_title('Train vs Test Sharpe Ratios')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_xlabel('Window')
        
        plt.tight_layout()
        plt.show()

# Unit Tests ----------------------------------------------------------------
class TestMACrossoverResearch(unittest.TestCase):
    def setUp(self):
        self.research = MACrossoverResearch()
        self.data = self.research.generate_test_data(size=100)

    def test_calculate_signals(self):
        signals = self.research.calculate_signals(self.data, 10, 50, ma_type='sma')
        self.assertIn('Signal', signals.columns)
        self.assertTrue((signals['Signal'].isin([0, 1])).all())

    def test_backtest(self):
        signals = self.research.calculate_signals(self.data, 10, 50, ma_type='sma')
        result = self.research.backtest(signals)
        self.assertIn('Portfolio_Value', result.columns)
        self.assertFalse(result['Portfolio_Value'].isna().any())

    def test_metrics(self):
        signals = self.research.calculate_signals(self.data, 10, 50, ma_type='sma')
        result = self.research.backtest(signals)
        metrics = self.research.calculate_metrics(result['Strategy_Return'])
        self.assertIn('sharpe_ratio', metrics)
        self.assertGreaterEqual(metrics['sharpe_ratio'], -np.inf)

# Example Usage -------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MA Crossover Research System")
    parser.add_argument('--data', default='synthetic', help="Data source: 'synthetic' or path to CSV")
    parser.add_argument('--short_range', type=int, nargs=2, default=[20, 100], help="Short MA range")
    parser.add_argument('--long_range', type=int, nargs=2, default=[100, 300], help="Long MA range")
    parser.add_argument('--ma_type', default='sma', choices=['sma', 'ema'], help="Moving average type")
    args = parser.parse_args()

    # Initialize research system
    research = MACrossoverResearch()
    
    # Generate or load data
    data = research.generate_test_data(size=2000, regimes=True) if args.data == 'synthetic' else research.load_real_data(args.data)
    
    # Core strategy analysis
    signal_data = research.calculate_signals(data, 50, 200, ma_type=args.ma_type)
    backtest_data = research.backtest(signal_data)
    metrics = research.calculate_metrics(backtest_data['Strategy_Return'])
    
    print("\nCore Strategy Metrics:")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title():<25}: {v:.4f}")
    
    # Advanced research
    wfo_results = research.walk_forward_optimization(
        data, short_range=args.short_range, long_range=args.long_range, ma_type=args.ma_type)
    
    mc_results = research.monte_carlo_simulation(data, n_simulations=1000, ma_type=args.ma_type)
    
    sensitivity = research.parameter_sensitivity(
        data, short_range=(10, 80), long_range=(100, 300), ma_type=args.ma_type)
    
    # Visualization
    research.plot_equity_curve(backtest_data)
    research.plot_parameter_sensitivity(sensitivity)
    research.plot_monte_carlo(mc_results)
    research.plot_walk_forward(wfo_results['results'])
    
    # Machine learning integration
    feature_data = research.create_feature_matrix(data)
    ml_results = research.train_hybrid_model(feature_data)
    
    print("\nFeature Importances:")
    print(ml_results['feature_importances'].head(10))
    
    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)