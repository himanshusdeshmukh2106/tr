"""
Complete pipeline runner for LSTM trading strategy
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

print("=" * 60)
print("LSTM Trading Strategy Pipeline")
print("=" * 60)

def run_data_preparation():
    """Step 1: Prepare data"""
    print("\n[1/4] Data Preparation")
    print("-" * 60)
    from src.data_preparation import main as prep_main
    prep_main()

def run_training_local():
    """Step 2: Train model locally"""
    print("\n[2/4] Model Training (Local)")
    print("-" * 60)
    from src.train_local import main as train_main
    train_main()

def run_training_gcp():
    """Step 2 (Alternative): Train model on GCP"""
    print("\n[2/4] Model Training (GCP)")
    print("-" * 60)
    from src.train_gcp import main as gcp_main
    gcp_main()

def run_backtest():
    """Step 3: Run backtest"""
    print("\n[3/4] Backtesting")
    print("-" * 60)
    print("Backtest will be run after training is complete.")
    print("Use: python src/backtest.py")

def main():
    """Run complete pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTM Trading Strategy Pipeline')
    parser.add_argument('--step', type=str, choices=['data', 'train', 'train-gcp', 'all'],
                       default='all', help='Pipeline step to run')
    
    args = parser.parse_args()
    
    try:
        if args.step in ['data', 'all']:
            run_data_preparation()
        
        if args.step in ['train', 'all']:
            run_training_local()
        
        if args.step == 'train-gcp':
            run_training_gcp()
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
