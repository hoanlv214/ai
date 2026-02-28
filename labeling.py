"""
labeling.py - Tao label du doan gia tang/giam

Ten cot dong theo PREDICT_MINUTES: 'future_return_15m', 'future_return_60m'...
Tat ca params doc tu config.py.
"""

import pandas as pd
import numpy as np
import gc
from config import CFG


def create_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tao label du doan gia. Params tu config.py:
    - CFG.PREDICT_BARS: so bars shift
    - CFG.LABEL_THRESHOLD: nguong phan loai
    - CFG.label_col(): ten cot dong theo PREDICT_MINUTES
    """
    n_bars = CFG.PREDICT_BARS
    threshold = CFG.LABEL_THRESHOLD
    col = CFG.label_col()  # 'future_return_15m', 'future_return_60m'...

    # Future return - in-place, ten cot dong
    df[col] = (
        df['close'].shift(-n_bars) / df['close'] - 1
    ).astype(np.float32)

    df.dropna(subset=[col], inplace=True)

    # Label
    if threshold == 0.0:
        df['y'] = (df[col] > 0).astype(np.int8)
    else:
        df['y'] = np.int8(-1)
        df.loc[df[col] > threshold, 'y'] = np.int8(1)
        df.loc[df[col] < -threshold, 'y'] = np.int8(0)

        rows_before = len(df)
        df = df[df['y'] >= 0].copy()
        print(f"   Dropped {rows_before - len(df)} neutral rows")

    df['y'] = df['y'].astype(np.int8)

    n1 = (df['y'] == 1).sum()
    n0 = (df['y'] == 0).sum()
    total = len(df)

    print(f"   Label col: '{col}' (shift={n_bars} bars = {CFG.PREDICT_MINUTES}m)")
    print(f"   Total: {total:,} | Tang: {n1:,} ({n1/total*100:.1f}%) | Giam: {n0:,} ({n0/total*100:.1f}%)")

    gc.collect()
    return df
