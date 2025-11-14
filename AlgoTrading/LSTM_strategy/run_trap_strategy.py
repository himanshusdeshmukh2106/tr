"""
Complete pipeline runner for EMA Trap Strategy
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

print("=" * 70)
print("EMA TRAP STRATEGY - Complete Pipeline")
print("=" * 70)

def step1_prepare_data():
    """Step 1: Download and prepare intraday data"""
    print("\n[1/3] Preparing Intraday Data")
    print("-" * 70)
    from src.intraday_data_prep import main as prep_main
    prep_main()

def step2_train_model():
    """Step 2: Train LSTM model"""
    print("\n[2/3] Training LSTM Model")
    print("-" * 70)
    print("Choose training option:")
    print("  1. Local training (free, slower)")
    print("  2. GCP training (paid, faster)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        from src.train_local import main as train_main
        train_main()
    elif choice == "2":
        from src.train_gcp import main as gcp_main
        gcp_main()
    else:
        print("Invalid choice. Skipping training.")

def step3_backtest():
    """Step 3: Run backtest"""
    print("\n[3/3] Running Backtest")
    print("-" * 70)
    from src.backtest_trap import main as backtest_main
    backtest_main()

def main():
    """Run complete pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EMA Trap Strategy Pipeline')
    parser.add_argument('--step', type=str, 
                       choices=['data', 'train', 'backtest', 'all'],
                       default='all', 
                       help='Pipeline step to run')
    parser.add_argument('--ticker', type=str, default='SPY',
                       help='Ticker symbol (e.g., SPY, RELIANCE.NS)')
    
    args = parser.parse_args()
    
    # Update ticker in config if provided
    if args.ticker:
        from config import config
        config.DATA_CONFIG["ticker"] = args.ticker
        print(f"\nUsing ticker: {args.ticker}")
    
    try:
        if args.step in ['data', 'all']:
            step1_prepare_data()
        
        if args.step in ['train', 'all']:
            step2_train_model()
        
        if args.step in ['backtest', 'all']:
            step3_backtest()
        
        print("\n" + "=" * 70)
        print("✓ Pipeline completed successfully!")
        print("=" * 70)
        print("\nResults saved in:")
        print("  - data/intraday_trap_data.csv")
        print("  - models/lstm_trading_best.h5")
        print("  - results/trap_strategy_backtest.png")
        print("  - results/trap_trades.csv")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
