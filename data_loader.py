"""
data_loader.py - Doc va load du lieu OHLCV da resample
Params doc tu config.py.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
import gc
from typing import Optional, List
from config import CFG


def load_resampled_parquet(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Doc du lieu OHLCV tu file parquet.
    file_path: None = dung CFG.DATA_PATH
    """
    path = file_path or CFG.DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Khong tim thay file: {path}")

    pf = pq.ParquetFile(path)
    file_size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"   File: {path}")
    print(f"   Size: {file_size_mb:.1f} MB")
    print(f"   Row groups: {pf.metadata.num_row_groups}")
    print(f"   Columns: {pf.schema.names}")

    available_cols = pf.schema.names

    needed_cols = ['open', 'high', 'low', 'close', 'volume']
    if 'timestamp' in available_cols and 'timestamp' not in needed_cols:
        needed_cols.append('timestamp')

    read_cols = [c for c in needed_cols if c in available_cols]

    table = pq.read_table(path, columns=read_cols)
    df = table.to_pandas()

    del table
    gc.collect()

    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Thieu cot '{col}' trong du lieu")

    if 'volume' not in df.columns:
        print("   WARN: Khong co cot 'volume', tao volume = 0")
        df['volume'] = np.float32(0) if CFG.USE_FLOAT32 else 0.0

    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ['timestamp', 'time', 'datetime']:
            if col in df.columns:
                df.index = pd.to_datetime(df[col], utc=True)
                df.drop(columns=[col], inplace=True)
                break
        else:
            raise ValueError("Index phai la DatetimeIndex hoac co cot timestamp")

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    df.sort_index(inplace=True)

    dup_mask = df.index.duplicated(keep='first')
    if dup_mask.any():
        print(f"   WARN: Drop {dup_mask.sum()} duplicate timestamps")
        df = df[~dup_mask]

    df = df[['open', 'high', 'low', 'close', 'volume']]

    if CFG.USE_FLOAT32:
        df = df.astype(np.float32)
        dtype_label = "float32"
    else:
        df = df.astype(np.float64)
        dtype_label = "float64"

    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"   Loaded {len(df):,} candles ({dtype_label})")
    print(f"   Memory: {mem_mb:.1f} MB")
    print(f"   Range: {df.index.min()} -> {df.index.max()}")
    print(f"   Price: {df['low'].min():.2f} - {df['high'].max():.2f}")

    gc.collect()
    return df
