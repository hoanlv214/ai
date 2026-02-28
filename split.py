"""
split.py - Chia du lieu train/test theo thoi gian
Params doc tu config.py.
"""

import pandas as pd
import gc
from typing import Tuple
from config import CFG


def time_series_split(df: pd.DataFrame,
                      train_end_date: str = None,
                      test_start_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chia du lieu theo thoi gian. Khong shuffle, khong overlap.
    Neu khong truyen dates, su dung CFG.TRAIN_END_DATE / CFG.TEST_START_DATE.
    Neu config cung None, tu dong split theo CFG.TRAIN_RATIO.
    """
    train_end_str = train_end_date or CFG.TRAIN_END_DATE
    test_start_str = test_start_date or CFG.TEST_START_DATE

    # Auto split neu khong co dates
    if train_end_str is None or test_start_str is None:
        total_days = (df.index.max() - df.index.min()).days
        split_dt = df.index.min() + pd.Timedelta(days=int(total_days * CFG.TRAIN_RATIO))
        train_end_str = split_dt.strftime('%Y-%m-%d')
        test_start_str = (split_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"   Auto split {CFG.TRAIN_RATIO:.0%}/{1-CFG.TRAIN_RATIO:.0%} at {train_end_str}")

    train_end = pd.Timestamp(train_end_str, tz='UTC')
    test_start = pd.Timestamp(test_start_str, tz='UTC')

    if train_end >= test_start:
        raise ValueError(f"train_end ({train_end_str}) phai < test_start ({test_start_str})")

    train_df = df.loc[:train_end]
    test_df = df.loc[test_start:]

    if len(train_df) == 0:
        raise ValueError(f"Train set trong! Data: {df.index.min()} - {df.index.max()}")
    if len(test_df) == 0:
        raise ValueError(f"Test set trong! Data: {df.index.min()} - {df.index.max()}")

    assert train_df.index.max() < test_df.index.min(), "Overlap!"

    gap = (test_df.index.min() - train_df.index.max()).days
    print(f"   Train: {len(train_df):,} ({train_df.index.min().strftime('%Y-%m-%d')} -> {train_df.index.max().strftime('%Y-%m-%d')})")
    print(f"   Test:  {len(test_df):,} ({test_df.index.min().strftime('%Y-%m-%d')} -> {test_df.index.max().strftime('%Y-%m-%d')})")
    print(f"   Gap: {gap} ngay | Ratio: {len(train_df)/len(test_df):.1f}")

    gc.collect()
    return train_df, test_df
