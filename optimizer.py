"""
optimizer.py - Optuna multi-timeframe optimizer

Tu dong tim bo (data_timeframe x predict_timeframe x LGB_params) toi uu nhat.

Output:
- Best combo: vd "1m_15m" = dung nen 1m de predict 15m
- Best LGB params
- Bang so sanh tat ca combos da test

Usage:
    python optimizer.py
    python optimizer.py --trials 100
    python optimizer.py --metric sharpe
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gc

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Can cai optuna: pip install optuna")
    sys.exit(1)

from config import CFG
from data_loader import load_resampled_parquet
from resampler import resample_ohlcv
from main import run_pipeline


# ============================================================
# DATA CACHE - Load 1 lan, resample on-the-fly
# ============================================================
_data_cache = {}


def get_data(bar_minutes: int) -> pd.DataFrame:
    """
    Lay OHLCV data cho 1 timeframe cu the.
    Cache ket qua de khong doc lai tu disk.
    Neu khong co file truc tiep, resample tu source nho nhat.

    Parameters
    ----------
    bar_minutes : int
        Timeframe can lay (1, 5, 15, 30, 60)

    Returns
    -------
    pd.DataFrame
        OHLCV data (KHONG DUOC MODIFY - luon .copy() truoc khi dung)
    """
    if bar_minutes in _data_cache:
        return _data_cache[bar_minutes]

    # Co file truc tiep?
    if bar_minutes in CFG.DATA_FILES and os.path.exists(CFG.DATA_FILES[bar_minutes]):
        print(f"   Loading {bar_minutes}m data from file...")
        CFG.DATA_PATH = CFG.DATA_FILES[bar_minutes]
        df = load_resampled_parquet()
        _data_cache[bar_minutes] = df
        return df

    # Khong co file -> resample tu source nho nhat co san
    available = sorted([tf for tf in CFG.DATA_FILES
                        if os.path.exists(CFG.DATA_FILES[tf])])

    if not available:
        raise FileNotFoundError("Khong tim thay bat ky data file nao!")

    # Chon source nho nhat de resample
    source_tf = available[0]
    source_df = get_data(source_tf)

    if bar_minutes <= source_tf:
        raise ValueError(
            f"Khong the resample {source_tf}m -> {bar_minutes}m "
            f"(can data nho hon {bar_minutes}m)")

    if bar_minutes % source_tf != 0:
        raise ValueError(
            f"{bar_minutes}m khong chia het cho {source_tf}m")

    print(f"   Resample {source_tf}m -> {bar_minutes}m on-the-fly...")
    df = resample_ohlcv(source_df, bar_minutes)
    _data_cache[bar_minutes] = df
    return df


def clear_cache():
    """Xoa data cache de giai phong memory."""
    global _data_cache
    _data_cache.clear()
    gc.collect()


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def create_objective(metric_name: str = 'auc'):
    """
    Tao objective function cho Optuna.

    Parameters
    ----------
    metric_name : str
        Metric de toi uu: 'auc', 'sharpe', 'winrate', 'f1'
    """

    def objective(trial: optuna.Trial) -> float:
        """Optuna trial: chon timeframe combo + LGB params, chay pipeline."""

        # === CHON TIMEFRAME ===
        bar_minutes = trial.suggest_categorical(
            'bar_minutes', CFG.CANDIDATE_DATA_TF)
        predict_minutes = trial.suggest_categorical(
            'predict_minutes', CFG.CANDIDATE_PREDICT_TF)

        # Validate combo
        if predict_minutes <= bar_minutes:
            # Khong co y nghia predict ngan hon data bar
            return float('-inf')
        if predict_minutes % bar_minutes != 0:
            # Predict phai chia het cho bar
            return float('-inf')

        # Update config
        CFG.update_timeframe(bar_minutes, predict_minutes)

        # === CHON LGB PARAMS (neu bat) ===
        if CFG.OPTUNA_TUNE_LGB:
            CFG.update_lgb_params(
                n_estimators=trial.suggest_int('n_estimators', 100, 800, step=100),
                max_depth=trial.suggest_int('max_depth', 3, 8),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                num_leaves=trial.suggest_int('num_leaves', 15, 63),
                min_child_samples=trial.suggest_int('min_child_samples', 20, 100, step=10),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            )

        # === CHAY PIPELINE ===
        combo = CFG.get_combo_name()
        print(f"\n{'='*50}")
        print(f"Trial {trial.number}: {combo}")
        print(f"  LGB: trees={CFG.LGB_N_ESTIMATORS} depth={CFG.LGB_MAX_DEPTH} "
              f"lr={CFG.LGB_LEARNING_RATE:.4f}")
        print(f"{'='*50}")

        try:
            # Lay data (tu cache hoac resample)
            data = get_data(bar_minutes)

            # Chay pipeline (verbose=False de khong in qua nhieu)
            results = run_pipeline(df=data, verbose=False)

            # Log metrics vao Optuna
            trial.set_user_attr('combo', combo)
            trial.set_user_attr('accuracy', results['accuracy'])
            trial.set_user_attr('auc', results['auc'])
            trial.set_user_attr('f1', results['f1'])
            trial.set_user_attr('sharpe', results['sharpe'])
            trial.set_user_attr('winrate', results['winrate'])
            trial.set_user_attr('n_trades', results['n_trades'])
            trial.set_user_attr('total_return', results['total_return'])
            trial.set_user_attr('max_drawdown', results['max_drawdown'])
            trial.set_user_attr('elapsed', results['elapsed'])

            score = results.get(metric_name, results['auc'])

            print(f"  => AUC={results['auc']:.4f} Sharpe={results['sharpe']:.4f} "
                  f"WR={results['winrate']:.2%} Score={score:.4f}")

            gc.collect()
            return score

        except Exception as e:
            print(f"  => FAILED: {e}")
            return float('-inf')

    return objective


# ============================================================
# RESULTS ANALYSIS
# ============================================================

def analyze_study(study: optuna.Study) -> pd.DataFrame:
    """
    Phan tich ket qua Optuna study va in bang so sanh.

    Returns
    -------
    pd.DataFrame
        Bang ket qua sorted theo score
    """
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if trial.value == float('-inf'):
            continue

        rows.append({
            'trial': trial.number,
            'combo': trial.user_attrs.get('combo', '?'),
            'bar_min': trial.params.get('bar_minutes', '?'),
            'pred_min': trial.params.get('predict_minutes', '?'),
            'score': trial.value,
            'auc': trial.user_attrs.get('auc', 0),
            'sharpe': trial.user_attrs.get('sharpe', 0),
            'winrate': trial.user_attrs.get('winrate', 0),
            'n_trades': trial.user_attrs.get('n_trades', 0),
            'total_ret': trial.user_attrs.get('total_return', 0),
            'max_dd': trial.user_attrs.get('max_drawdown', 0),
            'time': trial.user_attrs.get('elapsed', 0),
            'n_est': trial.params.get('n_estimators', ''),
            'depth': trial.params.get('max_depth', ''),
            'lr': trial.params.get('learning_rate', ''),
        })

    if not rows:
        print("Khong co trial nao hoan thanh!")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('score', ascending=False)

    print("\n" + "=" * 100)
    print("  OPTIMIZATION RESULTS - ALL COMPLETED TRIALS")
    print("=" * 100)
    print(f"{'#':<4} {'Combo':<10} {'Score':>8} {'AUC':>8} {'Sharpe':>8} "
          f"{'WR':>7} {'Trades':>7} {'Return':>9} {'MaxDD':>8} {'Time':>6}")
    print("-" * 100)

    for _, row in df.head(20).iterrows():
        print(f"{int(row['trial']):<4} {row['combo']:<10} "
              f"{row['score']:>8.4f} {row['auc']:>8.4f} {row['sharpe']:>8.4f} "
              f"{row['winrate']:>6.2%} {int(row['n_trades']):>7} "
              f"{row['total_ret']:>8.2%} {row['max_dd']:>7.2%} "
              f"{row['time']:>5.0f}s")

    print("-" * 100)

    # Best per combo
    print("\n  BEST PER COMBO:")
    best_per_combo = df.groupby('combo').first().sort_values('score', ascending=False)
    for combo, row in best_per_combo.iterrows():
        marker = " <<<< BEST" if combo == df.iloc[0]['combo'] else ""
        print(f"    {combo:<10} AUC={row['auc']:.4f} Sharpe={row['sharpe']:.4f} "
              f"WR={row['winrate']:.2%}{marker}")

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Optuna Multi-Timeframe Optimizer')
    parser.add_argument('--trials', type=int, default=None,
                        help=f'So trials (default: {CFG.OPTUNA_N_TRIALS})')
    parser.add_argument('--metric', type=str, default=None,
                        help=f'Metric toi uu (default: {CFG.OPTUNA_METRIC})')
    parser.add_argument('--no-tune-lgb', action='store_true',
                        help='Tat tune LGB params, chi tune timeframe')
    args = parser.parse_args()

    n_trials = args.trials or CFG.OPTUNA_N_TRIALS
    metric = args.metric or CFG.OPTUNA_METRIC

    if args.no_tune_lgb:
        CFG.OPTUNA_TUNE_LGB = False

    print("=" * 60)
    print("  OPTUNA MULTI-TIMEFRAME OPTIMIZER")
    print("=" * 60)
    print(f"  Trials:      {n_trials}")
    print(f"  Metric:      {metric}")
    print(f"  Tune LGB:    {CFG.OPTUNA_TUNE_LGB}")
    print(f"  Data TFs:    {CFG.CANDIDATE_DATA_TF}")
    print(f"  Predict TFs: {CFG.CANDIDATE_PREDICT_TF}")

    # Kiem tra data files
    print("\n  Data files:")
    for tf, path in CFG.DATA_FILES.items():
        exists = os.path.exists(path)
        status = "OK" if exists else "NOT FOUND"
        print(f"    {tf}m: {path} [{status}]")

    # Pre-load data
    print("\n  Pre-loading data...")
    for tf in CFG.CANDIDATE_DATA_TF:
        try:
            get_data(tf)
        except Exception as e:
            print(f"    WARN: Cannot load {tf}m data: {e}")

    # Tao Optuna study
    sampler = TPESampler(seed=CFG.RANDOM_STATE)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='btc_timeframe_optimizer',
    )

    # Chay
    t_start = time.time()

    study.optimize(
        create_objective(metric),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    total_time = time.time() - t_start

    # === KET QUA ===
    results_df = analyze_study(study)

    # Best trial
    best = study.best_trial
    print("\n" + "=" * 60)
    print("  BEST RESULT")
    print("=" * 60)
    print(f"  Combo:       {best.user_attrs.get('combo', '?')}")
    print(f"  Score:       {best.value:.4f} ({metric})")
    print(f"  AUC:         {best.user_attrs.get('auc', 0):.4f}")
    print(f"  Sharpe:      {best.user_attrs.get('sharpe', 0):.4f}")
    print(f"  Winrate:     {best.user_attrs.get('winrate', 0):.2%}")
    print(f"  N Trades:    {best.user_attrs.get('n_trades', 0)}")
    print(f"  Total Return:{best.user_attrs.get('total_return', 0):.2%}")
    print(f"  Max DD:      {best.user_attrs.get('max_drawdown', 0):.2%}")
    print(f"\n  Best params:")
    for key, val in best.params.items():
        print(f"    {key}: {val}")
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print("=" * 60)

    # Save results
    if len(results_df) > 0:
        out_path = os.path.join(os.path.dirname(__file__), "optimizer_results.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\n  Results saved to: {out_path}")

    # Cleanup
    clear_cache()


if __name__ == '__main__':
    main()
