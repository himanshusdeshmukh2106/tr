"""
Backtesting engine for EMA Trap Strategy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import TRADING_CONFIG, TRAP_STRATEGY_CONFIG, DATA_DIR, RESULTS_DIR
from src.trap_strategy import TrapStrategy


class TrapBacktester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.strategy = TrapStrategy()
        self.trades = []
        self.portfolio_values = []
        
    def run_backtest(self, df):
        """Run backtest on historical data"""
        print("Running backtest...")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        
        for idx in range(len(df)):
            timestamp = df.index[idx]
            current_price = df['Close'].iloc[idx]
            
            # Update portfolio value
            portfolio_value = self.capital
            if self.strategy.position:
                if self.strategy.position == 'LONG':
                    portfolio_value += (current_price - self.strategy.entry_price)
                else:  # SHORT
                    portfolio_value += (self.strategy.entry_price - current_price)
            
            self.portfolio_values.append({
                'timestamp': timestamp,
                'value': portfolio_value,
                'position': self.strategy.position
            })
            
            # Check if we have an open position
            if self.strategy.position:
                # Update trailing stop
                self.strategy.update_trailing_stop(current_price)
                
                # Check exit conditions
                should_exit, exit_reason = self.strategy.check_exit(current_price, timestamp)
                
                if should_exit:
                    # Calculate P&L
                    if self.strategy.position == 'LONG':
                        pnl = current_price - self.strategy.entry_price
                        pnl_pct = (pnl / self.strategy.entry_price) * 100
                    else:  # SHORT
                        pnl = self.strategy.entry_price - current_price
                        pnl_pct = (pnl / self.strategy.entry_price) * 100
                    
                    self.capital += pnl
                    
                    # Record trade
                    self.trades.append({
                        'entry_time': self.strategy.entry_price,
                        'exit_time': timestamp,
                        'entry_price': self.strategy.entry_price,
                        'exit_price': current_price,
                        'position': self.strategy.position,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason
                    })
                    
                    # Exit position
                    self.strategy.exit_position()
            
            # Check for new entry signals
            else:
                signal = self.strategy.generate_signal(df, idx)
                if signal:
                    self.strategy.enter_position(signal, current_price)
        
        # Close any open position at end
        if self.strategy.position:
            current_price = df['Close'].iloc[-1]
            if self.strategy.position == 'LONG':
                pnl = current_price - self.strategy.entry_price
            else:
                pnl = self.strategy.entry_price - current_price
            self.capital += pnl
        
        return self.calculate_metrics()

    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        trades_df = pd.DataFrame(self.trades)
        portfolio_df = pd.DataFrame(self.portfolio_values)
        
        if len(trades_df) == 0:
            print("No trades executed!")
            return None, None, None
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Returns
        final_capital = self.capital
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Sharpe ratio
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
        sharpe_ratio = (portfolio_df['returns'].mean() / portfolio_df['returns'].std()) * np.sqrt(252 * 78)  # 78 5-min candles per day
        
        # Drawdown
        portfolio_df['cummax'] = portfolio_df['value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['cummax']) / portfolio_df['cummax'] * 100
        max_drawdown = portfolio_df['drawdown'].min()
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': final_capital
        }
        
        return metrics, portfolio_df, trades_df
    
    def print_results(self, metrics):
        """Print backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS - EMA TRAP STRATEGY")
        print("="*60)
        print(f"\nCapital:")
        print(f"  Initial: ${self.initial_capital:,.2f}")
        print(f"  Final:   ${metrics['final_capital']:,.2f}")
        print(f"  P&L:     ${metrics['total_pnl']:,.2f}")
        print(f"  Return:  {metrics['total_return']:.2f}%")
        
        print(f"\nTrades:")
        print(f"  Total:   {metrics['total_trades']}")
        print(f"  Winners: {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
        print(f"  Losers:  {metrics['losing_trades']}")
        
        print(f"\nPerformance:")
        print(f"  Avg Win:       ${metrics['avg_win']:.2f}")
        print(f"  Avg Loss:      ${metrics['avg_loss']:.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:  {metrics['max_drawdown']:.2f}%")
        print("="*60)
    
    def plot_results(self, portfolio_df, trades_df):
        """Plot backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Portfolio value
        axes[0].plot(portfolio_df['timestamp'], portfolio_df['value'], linewidth=2)
        axes[0].axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        axes[1].fill_between(portfolio_df['timestamp'], portfolio_df['drawdown'], 0, 
                            alpha=0.3, color='red')
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Trade P&L distribution
        axes[2].hist(trades_df['pnl_pct'], bins=30, alpha=0.7, edgecolor='black')
        axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[2].set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('P&L (%)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'trap_strategy_backtest.png', dpi=300, bbox_inches='tight')
        print(f"\nResults saved to {RESULTS_DIR / 'trap_strategy_backtest.png'}")
        plt.show()


def main():
    """Run backtest"""
    # Load data
    df = pd.read_csv(DATA_DIR / "intraday_trap_data.csv", index_col=0, parse_dates=True)
    
    print(f"Loaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Run backtest
    backtester = TrapBacktester(initial_capital=100000)
    metrics, portfolio_df, trades_df = backtester.run_backtest(df)
    
    if metrics:
        # Print results
        backtester.print_results(metrics)
        
        # Save results
        trades_df.to_csv(RESULTS_DIR / 'trap_trades.csv', index=False)
        portfolio_df.to_csv(RESULTS_DIR / 'trap_portfolio.csv', index=False)
        
        # Plot results
        backtester.plot_results(portfolio_df, trades_df)


if __name__ == "__main__":
    main()
