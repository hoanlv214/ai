"""
model.py - Train va predict voi LightGBM
Tat ca hyperparams doc tu config.py.

Early stopping: tach 10% cuoi cua train lam validation set.
Giu thu tu thoi gian (khong shuffle) de tranh data leakage.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import gc
from config import CFG


def train_model(X_train: pd.DataFrame,
                y_train: pd.Series) -> lgb.LGBMClassifier:
    """
    Train LightGBM classifier voi early stopping.
    Tach 10% cuoi train lam validation (giu thu tu thoi gian).
    """
    model = lgb.LGBMClassifier(
        n_estimators=CFG.LGB_N_ESTIMATORS,
        max_depth=CFG.LGB_MAX_DEPTH,
        learning_rate=CFG.LGB_LEARNING_RATE,
        num_leaves=CFG.LGB_NUM_LEAVES,
        min_child_samples=CFG.LGB_MIN_CHILD_SAMPLES,
        subsample=CFG.LGB_SUBSAMPLE,
        colsample_bytree=CFG.LGB_COLSAMPLE,
        is_unbalance=CFG.LGB_IS_UNBALANCE,
        random_state=CFG.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    # Tach 10% cuoi train lam validation (giu thu tu thoi gian, KHONG shuffle)
    val_size = max(1, int(len(X_train) * 0.1))
    split_idx = len(X_train) - val_size

    X_tr = X_train.iloc[:split_idx]
    y_tr = y_train.iloc[:split_idx]
    X_val = X_train.iloc[split_idx:]
    y_val = y_train.iloc[split_idx:]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # Metrics
    train_acc = (model.predict(X_tr) == y_tr).mean()
    val_acc = (model.predict(X_val) == y_val).mean()
    best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators

    print(f"   {best_iter}/{CFG.LGB_N_ESTIMATORS} trees (early stop), "
          f"depth={CFG.LGB_MAX_DEPTH}, lr={CFG.LGB_LEARNING_RATE}")
    print(f"   Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
    print(f"   Features: {X_train.shape[1]}, Samples: {X_train.shape[0]:,} "
          f"(train={len(X_tr):,}, val={len(X_val):,})")

    # Kiem tra overfit
    overfit_gap = train_acc - val_acc
    if overfit_gap > 0.05:
        print(f"   WARN: Possible overfit! Gap={overfit_gap:.4f}")

    del X_tr, y_tr, X_val, y_val
    gc.collect()
    return model


def predict_proba(model: lgb.LGBMClassifier,
                  X_test: pd.DataFrame) -> np.ndarray:
    """Du doan xac suat tang gia."""
    probas = model.predict_proba(X_test)[:, 1].astype(np.float32)

    t = CFG.PROB_THRESHOLD
    print(f"   Predictions: {len(probas):,}")
    print(f"   Mean: {probas.mean():.4f}, Range: [{probas.min():.4f}, {probas.max():.4f}]")
    print(f"   Prob > {t}: {(probas > t).sum():,} ({(probas > t).mean()*100:.1f}%)")

    return probas


def get_feature_importance(model: lgb.LGBMClassifier,
                           feature_names: list) -> pd.DataFrame:
    """Lay feature importance tu model."""
    imp = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    imp['pct'] = imp['importance'] / imp['importance'].sum() * 100
    return imp
