"""
main.py - Pipeline du doan UP/DOWN

Muc tieu: Nhan nen 5m hien tai -> Du doan 15m sau UP hay DOWN, xac suat bao nhieu %

Output chinh:
- AUC, Accuracy
- Probability Binning: khi model noi X%, thuc te dung bao nhieu %?
- Calibration: model co dang tin khong?

Usage:
    python main.py
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
    Train model va danh gia kha nang du doan UP/DOWN.

    Returns
    -------
    dict: accuracy, auc, f1, winrate tai cac muc xac suat
    """
    _print = print if verbose else lambda *a, **k: None

    t_start = time.time()

    if verbose:
        CFG.print_config()
        print()

    # [1/7] Load data
    _print("[1/7] Load data...")
    if df is None:
        df = load_resampled_parquet()
    else:
        df = df.copy()
    _print(f"   {len(df):,} candles loaded")

    # [2/7] Features
    _print("[2/7] Create features...")
    df = create_features(df)

    # [3/7] Labels
    _print("[3/7] Create labels...")
    df = create_label(df)

    # [4/7] Split
    _print("[4/7] Split...")
    train_df, test_df = time_series_split(df)
    del df
    gc.collect()

    # [5/7] Prepare X, y
    _print("[5/7] Prepare X, y...")
    feature_cols = get_feature_columns()
    X_train = train_df[feature_cols]
    y_train = train_df['y']
    X_test = test_df[feature_cols]
    y_test = test_df['y']
    _print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")

    # [6/7] Train
    _print("[6/7] Train LightGBM...")
    lgb_model = train_model(X_train, y_train)

    if verbose:
        imp = get_feature_importance(lgb_model, feature_cols)
        print("\n   Top features:")
        for _, row in imp.head(7).iterrows():
            print(f"     {row['feature']:<25} {row['pct']:.1f}%")

    del X_train, y_train, train_df
    gc.collect()

    # [7/7] Predict & Evaluate
    _print("\n[7/7] Predict & Evaluate...")
    y_proba = predict_proba(lgb_model, X_test)
    y_pred = (y_proba > 0.5).astype(np.int8)

    metrics = evaluate_model(y_test.values, y_pred, y_proba)

    if verbose:
        print()

    bin_df = probability_binning(y_test.values, y_proba)

    # === Calibration check ===
    if verbose:
        print()
        print("=" * 60)
        print("  CALIBRATION CHECK")
        print("=" * 60)
        print("  Model noi X% -> thuc te dung bao nhieu %?")
        print()
        for _, row in bin_df.iterrows():
            if row['count'] > 0:
                model_prob = row['avg_prob'] * 100
                actual_wr = row['winrate'] * 100
                diff = actual_wr - model_prob
                status = "✓" if abs(diff) < 3 else "⚠"
                print(f"    Model noi {model_prob:5.1f}% UP -> Thuc te {actual_wr:5.1f}% "
                      f"(sai lech {diff:+.1f}%) {status}")

    elapsed = time.time() - t_start

    # === KET QUA ===
    results = {
        'combo_name': CFG.get_combo_name(),
        'bar_minutes': CFG.BAR_MINUTES,
        'predict_minutes': CFG.PREDICT_MINUTES,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc'],
        'elapsed': elapsed,
    }

    # Them winrate tai cac muc xac suat
    for _, row in bin_df.iterrows():
        if row['count'] > 0:
            key = f"wr_{row['bin'].replace(' ', '').replace('-', '_')}"
            results[key] = row['winrate']
            results[f"n_{row['bin'].replace(' ', '').replace('-', '_')}"] = int(row['count'])

    if verbose:
        print()
        print("=" * 60)
        print(f"  FINAL RESULTS [{results['combo_name']}]")
        print("=" * 60)
        print(f"   AUC:        {results['auc']:.4f}")
        print(f"   Accuracy:   {results['accuracy']:.4f}")
        print(f"   F1:         {results['f1']:.4f}")
        print(f"   Time:       {elapsed:.1f}s")
        print()
        print("   Khi dung model de du doan:")
        print(f"     - Model noi UP (>50%): dung {metrics['precision']:.1%}")
        print(f"     - Model noi UP >55%:   dung ~{bin_df[bin_df['bin'].str.contains('0.55')]['winrate'].values[0]:.1%}" if len(bin_df[bin_df['bin'].str.contains('0.55')]) > 0 else "")
        print(f"     - Model noi DOWN (<45%): dung ~{100-bin_df[bin_df['bin'].str.contains('0.40')]['winrate'].values[0]*100:.1f}%" if len(bin_df[bin_df['bin'].str.contains('0.40')]) > 0 else "")
        print("=" * 60)

    del test_df, X_test, y_test, y_proba, y_pred, lgb_model
    gc.collect()

    return results


def main():
    run_pipeline(verbose=True)


if __name__ == '__main__':
    main()
