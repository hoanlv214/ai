"""
main.py - Pipeline chinh

2 che do:
1. Chay don: python main.py  (dung config hien tai)
2. Goi tu optimizer: run_pipeline(df) tra ve metrics dict
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import gc

from config import CFG
from data_loader import load_resampled_parquet
from features import create_features, get_feature_columns
from labeling import create_label
from split import time_series_split
from model import train_model, predict_proba, get_feature_importance
from evaluate import evaluate_model, probability_binning
from backtest import run_backtest


def print_memory(label: str = ""):
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info().rss / (1024 ** 2)
        print(f"   [MEM] {label}: {mem:.0f} MB")
    except ImportError:
        pass


def run_pipeline(df: pd.DataFrame = None,
                 verbose: bool = True) -> dict:
    """
    Chay pipeline ML va tra ve metrics dict.

    Parameters
    ----------
    df : pd.DataFrame, optional
        OHLCV data. None = load tu CFG.DATA_PATH
    verbose : bool
        True = in chi tiet, False = silent (cho optimizer)

    Returns
    -------
    dict
        Ket qua: accuracy, auc, f1, sharpe, max_drawdown, winrate, n_trades,
                 total_return, combo_name
    """
    _print = print if verbose else lambda *a, **k: None

    t_start = time.time()

    if verbose:
        CFG.print_config()
        print()

    # [1] Load data
    _print(f"[1/9] Load data...")
    if df is None:
        df = load_resampled_parquet()
    else:
        df = df.copy()  # Khong modify data goc (quan trong cho optimizer cache)
    _print(f"   {len(df):,} candles loaded")

    # [2] Features
    _print("[2/9] Create features...")
    df = create_features(df)

    # [3] Labels
    _print("[3/9] Create labels...")
    df = create_label(df)

    # [4] Split
    _print("[4/9] Split...")
    train_df, test_df = time_series_split(df)

    del df
    gc.collect()

    # [5] Prepare X, y
    _print("[5/9] Prepare X, y...")
    feature_cols = get_feature_columns()

    X_train = train_df[feature_cols]
    y_train = train_df['y']
    X_test = test_df[feature_cols]
    y_test = test_df['y']

    _print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")

    # [6] Train
    _print("[6/9] Train LightGBM...")
    lgb_model = train_model(X_train, y_train)

    if verbose:
        imp = get_feature_importance(lgb_model, feature_cols)
        print("\n   Top features:")
        for _, row in imp.head(5).iterrows():
            print(f"     {row['feature']:<25} {row['pct']:.1f}%")

    del X_train, y_train, train_df
    gc.collect()

    # [7] Predict
    _print("[7/9] Predict...")
    y_proba = predict_proba(lgb_model, X_test)
    y_pred = (y_proba > 0.5).astype(np.int8)

    # [8] Evaluate
    _print("[8/9] Evaluate...")
    metrics = evaluate_model(y_test.values, y_pred, y_proba)

    if verbose:
        print()
        probability_binning(y_test.values, y_proba)

    # [9] Backtest
    _print("[9/9] Backtest...")
    label_col = CFG.label_col()  # 'future_return_15m', 'future_return_60m'...
    bt = run_backtest(
        y_proba=y_proba,
        future_returns=test_df[label_col],
    )

    # Winrate tai prob threshold - dung ACTUAL RETURN (nhat quan voi backtest)
    t = CFG.PROB_THRESHOLD
    high_conf_mask = y_proba > t
    if high_conf_mask.any():
        actual_returns = test_df[label_col].values[high_conf_mask]
        winrate_at_threshold = float((actual_returns > 0).mean())
    else:
        winrate_at_threshold = 0.0

    elapsed = time.time() - t_start

    # === KET QUA TONG HOP ===
    results = {
        'combo_name': CFG.get_combo_name(),
        'bar_minutes': CFG.BAR_MINUTES,
        'predict_minutes': CFG.PREDICT_MINUTES,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc'],
        'sharpe': bt['sharpe'],
        'max_drawdown': bt['max_drawdown'],
        'winrate': winrate_at_threshold,
        'n_trades': bt['n_trades'],
        'total_return': bt['total_return'],
        'bnh_return': bt['bnh_return'],
        'elapsed': elapsed,
    }

    if verbose:
        print()
        print("=" * 60)
        print(f"  FINAL RESULTS [{results['combo_name']}]")
        print("=" * 60)
        print(f"   Accuracy:     {results['accuracy']:.4f}")
        print(f"   AUC:          {results['auc']:.4f}")
        print(f"   Sharpe:       {results['sharpe']:.4f}")
        print(f"   Max DD:       {results['max_drawdown']:.2%}")
        print(f"   Winrate>{t}:  {results['winrate']:.2%} ({results['n_trades']} trades)")
        print(f"   Strategy:     {results['total_return']:.2%}")
        print(f"   Buy&Hold:     {results['bnh_return']:.2%}")
        print(f"   Time:         {elapsed:.1f}s")
        print("=" * 60)

    # Cleanup
    del test_df, X_test, y_test, y_proba, y_pred, lgb_model
    gc.collect()

    return results


def main():
    """Chay pipeline don voi config hien tai."""
    run_pipeline(verbose=True)


if __name__ == '__main__':
    main()
