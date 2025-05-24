# Moving Average Crossover Research Project

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

Comprehensive research framework for analyzing moving average crossover strategies in financial markets.

## Features

- Multiple MA strategy implementations
- Walk-forward optimization
- Monte Carlo simulation
- Machine learning integration
- Advanced visualization

## Installation

```bash
git clone https://github.com/yourusername/MA_Crossover_Research.git
cd MA_Crossover_Research
pip install -r requirements.txt
```


Usage
from src.strategy import MACrossover
from src.backtest import Backtester

# Initialize strategy
strategy = MACrossover(short_window=50, long_window=200)

# Run backtest
backtester = Backtester(strategy)
results = backtester.run(data)
