"""
Backtesting engine for LSTM trading strategy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import TRADING_CONFIG, MODELS_DIR, RESULTS_DIR


class Backtester:
    def __init__(self, model_path, initial_capital=100000):
        self.model = keras.models.load_model(model_path)
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []
        self.portfolio_values = []
        
    def generate_signals(self, X):
        """Generate trading signals from model predictions"""
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def backtest(self, X, prices, dates, threshold=0.5):
        """
        Run backtest
        threshold: probability threshold for buy signal
        """
        signals = self.generate_signals(X)
        
        for i in range(len(signals)):
            current_price = prices[i]
            signal_prob = signals[i]
            
            # Buy signal
            if signal_prob > threshold and self.position == 0:
                shares = (self.capital * TRADING_CONFIG["position_size"]) / current_price
                cost = shares * current_price * (1 + TRADING_CONFIG["transaction_cost"])
                
                if cost <= self.capital:
                    self.position = shares
                    self.capital -= cost
                    self.trades.append({
                        'date': dates[i],
                        'type': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'signal_prob': signal_prob
                    })
            
            # Sell signal
            elif signal_prob < (1 - threshold) and self.position > 0:
                proceeds = self.position * current_price * (1 - TRADING_CONFIG["transaction_cost"])
                self.capital += proceeds
                
                self.trades.append({
                    'date': dates[i],
                    'type': 'SELL',
                    'price': current_price,
                    'shares': self.position,
                    'signal_prob': signal_prob
                })
                self.position = 0
            
            # Calculate portfolio value
            portfolio_value = self.capital + (self.position * current_price)
            self.portfolio_values.append({
                'date': dates[i],
                'value': portfolio_value,
                'returns': (portfolio_value / self.initial_capital - 1) * 100
            })
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        df = pd.DataFrame(self.portfolio_values)
        
        final_value = df['value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        df['daily_returns'] = df['value'].pct_change()
        sharpe_ratio = (df['daily_returns'].mean() / df['daily_returns'].std()) * np.sqrt(252)
        
        df['cummax'] = df['value'].cummax()
        df['drawdown'] = (df['value'] - df['cummax']) / df['cummax'] * 100
        max_drawdown = df['drawdown'].min()
        
        trades_df = pd.DataFrame(self.trades)
        num_trades = len(trades_df)
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'final_value': final_value
        }
        
        return metrics, df, trades_df
    
    def plot_results(self, portfolio_df, benchmark_prices=None):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Portfolio value
        axes[0].plot(portfolio_df['date'], portfolio_df['value'], label='Strategy', linewidth=2)
        if benchmark_prices is not None:
            axes[0].plot(portfolio_df['date'], benchmark_prices, label='Buy & Hold', linewidth=2, alpha=0.7)
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        axes[1].fill_between(portfolio_df['date'], portfolio_df['drawdown'], 0, alpha=0.3, color='red')
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'backtest_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {RESULTS_DIR / 'backtest_results.png'}")
        plt.show()
