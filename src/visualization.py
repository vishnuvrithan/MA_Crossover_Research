"""
visualization.py: Plot equity curves, heatmaps, simulation results
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_equity_curve(data: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(data['Cumulative_Market'], label='Market')
    plt.plot(data['Cumulative_Strategy'], label='Strategy')
    plt.legend()
    plt.title('Equity Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_parameter_sensitivity(heatmap_data, short_range, long_range):
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, xticklabels=long_range, yticklabels=short_range, cmap='coolwarm')
    plt.title('Parameter Sensitivity Heatmap')
    plt.xlabel('Long MA')
    plt.ylabel('Short MA')
    plt.tight_layout()
    plt.show()


def plot_monte_carlo(simulation_results):
    plt.figure(figsize=(10, 5))
    sns.histplot(simulation_results, kde=True)
    plt.title('Monte Carlo Simulation Results')
    plt.xlabel('Final Portfolio Value')
    plt.tight_layout()
    plt.show()