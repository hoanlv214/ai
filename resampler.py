"""
resampler.py - Module resample tickdata/OHLCV

Functions:
- resample_to_1m(): tickdata -> nen 1 phut
- resample_to_5m(): tickdata -> nen 5 phut
- resample_ohlcv(): OHLCV nen nho -> OHLCV nen lon (vd: 1m -> 15m, 5m -> 1h)
"""

import pandas as pd
import numpy as np


def resample_ohlcv(df: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
    """
    Resample OHLCV data tu nen nho sang nen lon hon.
    Vi du: 1m -> 5m, 5m -> 1h, 1m -> 15m...

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (columns: open, high, low, close, volume)
    target_minutes : int
        Timeframe dich (phut). Vd: 5, 15, 30, 60

    Returns
    -------
    pd.DataFrame
        OHLCV da resample, dropna, khong forward fill
    """
    rule = f'{target_minutes}min'

    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Drop nen khong co du lieu (khong forward fill)
    resampled.dropna(inplace=True)

    # Giu nguyen dtype
    resampled = resampled.astype(df.dtypes.to_dict())

    print(f"   Resample: {len(df):,} bars -> {len(resampled):,} bars ({target_minutes}m)")

    return resampled


def resample_to_1m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample tickdata thanh nen 1 phut."""
    if 'price' not in df.columns:
        raise ValueError("DataFrame phai co cot 'price'")

    ohlc = df['price'].resample('1min').ohlc()
    if 'quantity' in df.columns:
        ohlc['volume'] = df['quantity'].resample('1min').sum()
    else:
        ohlc['volume'] = 0.0

    ohlc.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    return ohlc


def resample_to_5m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample tickdata thanh nen 5 phut."""
    if 'price' not in df.columns:
        raise ValueError("DataFrame phai co cot 'price'")

    ohlc = df['price'].resample('5min').ohlc()
    if 'quantity' in df.columns:
        ohlc['volume'] = df['quantity'].resample('5min').sum()
    else:
        ohlc['volume'] = 0.0

    ohlc.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    return ohlc
