"""
features.py - Module tao technical indicators va features cho ML model

Tat ca features chi su dung du lieu qua khu (khong data leakage).
Tat ca window sizes doc tu config.py.
"""

import pandas as pd
import numpy as np
import gc
from config import CFG


# ============================================================
# PRICE FEATURES
# ============================================================

def _log_return(close: pd.Series) -> pd.Series:
    """Log return 1 bar: ln(close_t / close_{t-1})"""
    return np.log(close / close.shift(1))


def _return_short(close: pd.Series) -> pd.Series:
    """Return ngan = RETURN_SHORT_WINDOW bars (5 phut)"""
    return close.pct_change(CFG.RETURN_SHORT_WINDOW)


def _return_long(close: pd.Series) -> pd.Series:
    """Return dai = RETURN_LONG_WINDOW bars (15 phut)"""
    return close.pct_change(CFG.RETURN_LONG_WINDOW)


def _rolling_vol_short(log_ret: pd.Series) -> pd.Series:
    """Rolling volatility ngan = VOL_SHORT_WINDOW bars"""
    w = CFG.VOL_SHORT_WINDOW
    return log_ret.rolling(window=w, min_periods=w).std()


def _rolling_vol_long(log_ret: pd.Series) -> pd.Series:
    """Rolling volatility dai = VOL_LONG_WINDOW bars"""
    w = CFG.VOL_LONG_WINDOW
    return log_ret.rolling(window=w, min_periods=w).std()


# ============================================================
# VOLUME FEATURES
# ============================================================

def _volume_zscore(volume: pd.Series) -> pd.Series:
    """Volume z-score over VOLUME_ZSCORE_WINDOW bars."""
    w = CFG.VOLUME_ZSCORE_WINDOW
    rm = volume.rolling(window=w, min_periods=w).mean()
    rs = volume.rolling(window=w, min_periods=w).std()
    rs = rs.replace(0, np.nan)
    return (volume - rm) / rs


def _volume_change(volume: pd.Series) -> pd.Series:
    """Volume change: volume_t / volume_{t-1} - 1"""
    return volume.pct_change(1)


# ============================================================
# TECHNICAL INDICATORS - Tu tinh bang pandas/numpy
# ============================================================

def _rsi(close: pd.Series) -> pd.Series:
    """RSI - Wilder's smoothing. Period tu config."""
    p = CFG.RSI_PERIOD
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / p, min_periods=p, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / p, min_periods=p, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_histogram(close: pd.Series) -> pd.Series:
    """MACD Histogram. Params tu config."""
    f, s, sig = CFG.MACD_FAST, CFG.MACD_SLOW, CFG.MACD_SIGNAL
    ema_f = close.ewm(span=f, min_periods=f, adjust=False).mean()
    ema_s = close.ewm(span=s, min_periods=s, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, min_periods=sig, adjust=False).mean()
    return macd - signal


def _bollinger_band_width(close: pd.Series) -> pd.Series:
    """BB Width. Params tu config."""
    p, n = CFG.BB_PERIOD, CFG.BB_STD
    sma = close.rolling(window=p, min_periods=p).mean()
    std = close.rolling(window=p, min_periods=p).std()
    upper = sma + n * std
    lower = sma - n * std
    return (upper - lower) / sma.replace(0, np.nan)


# ============================================================
# CANDLE PATTERN FEATURES - numpy vectorized
# ============================================================

def _candle_body_ratio(o, h, l, c):
    body = np.abs(c - o)
    total = h - l
    total = np.where(total == 0, np.nan, total)
    return body / total


def _upper_wick_ratio(o, h, l, c):
    top = np.maximum(o, c)
    wick = h - top
    total = h - l
    total = np.where(total == 0, np.nan, total)
    return wick / total


def _lower_wick_ratio(o, h, l, c):
    bottom = np.minimum(o, c)
    wick = bottom - l
    total = h - l
    total = np.where(total == 0, np.nan, total)
    return wick / total


# ============================================================
# MAIN
# ============================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tao tat ca features tu OHLCV. In-place, toi uu memory.
    Tat ca params doc tu config.py.
    """
    rows_before = len(df)

    close = df['close']
    log_ret = _log_return(close)

    df['log_return_1m'] = log_ret
    df['return_5m'] = _return_short(close)
    df['return_15m'] = _return_long(close)
    df['rolling_vol_15m'] = _rolling_vol_short(log_ret)
    df['rolling_vol_60m'] = _rolling_vol_long(log_ret)

    del log_ret

    df['volume_zscore_15m'] = _volume_zscore(df['volume'])
    df['volume_change'] = _volume_change(df['volume'])

    df['rsi_14'] = _rsi(df['close'])
    df['macd_histogram'] = _macd_histogram(df['close'])
    df['bb_width'] = _bollinger_band_width(df['close'])

    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values

    df['candle_body_ratio'] = _candle_body_ratio(o, h, l, c)
    df['upper_wick_ratio'] = _upper_wick_ratio(o, h, l, c)
    df['lower_wick_ratio'] = _lower_wick_ratio(o, h, l, c)

    del o, h, l, c

    df.dropna(inplace=True)
    rows_after = len(df)

    if CFG.USE_FLOAT32:
        for col in get_feature_columns():
            df[col] = df[col].astype(np.float32)

    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"   Features: {len(get_feature_columns())}")
    print(f"   Dropped {rows_before - rows_after} NaN ({rows_before:,} -> {rows_after:,})")
    print(f"   Memory: {mem_mb:.1f} MB")

    gc.collect()
    return df


def get_feature_columns() -> list:
    """Tra ve danh sach ten cac cot feature."""
    return [
        'log_return_1m',
        'return_5m',
        'return_15m',
        'rolling_vol_15m',
        'rolling_vol_60m',
        'volume_zscore_15m',
        'volume_change',
        'rsi_14',
        'macd_histogram',
        'bb_width',
        'candle_body_ratio',
        'upper_wick_ratio',
        'lower_wick_ratio',
    ]
