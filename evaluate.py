"""
evaluate.py - Danh gia model performance
Probability bins doc tu config.py.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from typing import Dict
from config import CFG


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   y_proba: np.ndarray) -> Dict[str, float]:
    """Danh gia toan dien model."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba),
    }
    cm = confusion_matrix(y_true, y_pred)

    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    print(f"   AUC:       {metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"                   FN={cm[1][0]}  TP={cm[1][1]}")
    print(classification_report(y_true, y_pred, target_names=['Giam(0)', 'Tang(1)']))
    return metrics


def probability_binning(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    """Chia xac suat thanh cac bin va tinh winrate. Bins tu config.py."""
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})

    results = []
    for low, high in CFG.PROB_BINS:
        if high > 1.0:
            label = f"> {low:.2f}"
            mask = df['y_proba'] >= low
        else:
            label = f"{low:.2f}-{high:.2f}"
            mask = (df['y_proba'] >= low) & (df['y_proba'] < high)

        bd = df[mask]
        wr = bd['y_true'].mean() if len(bd) > 0 else 0.0
        ap = bd['y_proba'].mean() if len(bd) > 0 else 0.0
        results.append({
            'bin': label, 'count': len(bd),
            'pct': len(bd) / len(df) * 100 if len(df) > 0 else 0,
            'winrate': wr, 'avg_prob': ap,
        })

    result_df = pd.DataFrame(results)

    print("=" * 60)
    print("PROBABILITY BINNING - WINRATE BY CONFIDENCE")
    print("=" * 60)
    print(f"{'Bin':<12} {'Count':>8} {'%Total':>7} {'Winrate':>9} {'AvgProb':>9}")
    print("-" * 50)
    for _, row in result_df.iterrows():
        print(f"{row['bin']:<12} {int(row['count']):>8} {row['pct']:>6.1f}% "
              f"{row['winrate']:>8.2%} {row['avg_prob']:>8.4f}")

    t = CFG.PROB_THRESHOLD
    high_conf = df[df['y_proba'] > t]
    if len(high_conf) > 0:
        print(f"\nWinrate khi prob > {t}: {high_conf['y_true'].mean():.2%} "
              f"({len(high_conf):,} trades)")
    else:
        print(f"\nKhong co prediction nao > {t}")

    return result_df
