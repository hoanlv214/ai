"""
features.py - Technical indicators va features cho ML model

Features chia thanh 4 nhom:
1. PRICE: return, volatility
2. VOLUME: OBV, MFI, RVOL, A/D, volume-price interaction
3. TECHNICAL: RSI, MACD, BB
4. CANDLE: body ratio, wicks

Tat ca chi dung du lieu qua khu. Params tu config.py.
"""

import pandas as pd
import numpy as np
import gc
from config import CFG


# ============================================================
# 1. PRICE FEATURES
# ============================================================

def _log_return(close: pd.Series) -> pd.Series:
    """Log return 1 bar"""
    return np.log(close / close.shift(1))

def _return_short(close: pd.Series) -> pd.Series:
    """Return ngan (5 phut)"""
    return close.pct_change(CFG.RETURN_SHORT_WINDOW)

def _return_long(close: pd.Series) -> pd.Series:
    """Return dai (15 phut)"""
    return close.pct_change(CFG.RETURN_LONG_WINDOW)

def _rolling_vol_short(log_ret: pd.Series) -> pd.Series:
    """Volatility ngan"""
    w = CFG.VOL_SHORT_WINDOW
    return log_ret.rolling(window=w, min_periods=w).std()

def _rolling_vol_long(log_ret: pd.Series) -> pd.Series:
    """Volatility dai"""
    w = CFG.VOL_LONG_WINDOW
    return log_ret.rolling(window=w, min_periods=w).std()


# ============================================================
# 2. VOLUME FEATURES - Nhom features moi, khai thac volume sau
# ============================================================

def _volume_zscore(volume: pd.Series) -> pd.Series:
    """Volume z-score: volume bat thuong khong?"""
    w = CFG.VOLUME_ZSCORE_WINDOW
    rm = volume.rolling(window=w, min_periods=w).mean()
    rs = volume.rolling(window=w, min_periods=w).std()
    rs = rs.replace(0, np.nan)
    return (volume - rm) / rs

def _volume_change(volume: pd.Series) -> pd.Series:
    """Volume change vs bar truoc"""
    return volume.pct_change(1)

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On Balance Volume (OBV).
    - Gia tang: cong volume
    - Gia giam: tru volume
    OBV tang = tien dang chay vao, OBV giam = tien dang rut ra.
    """
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()

def _obv_slope(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    OBV slope (rate of change) over VOL_SHORT_WINDOW.
    OBV tang nhanh = ap luc mua manh.
    OBV giam nhanh = ap luc ban manh.
    """
    obv = _obv(close, volume)
    w = CFG.VOL_SHORT_WINDOW
    return obv.diff(w) / (obv.rolling(w).mean().replace(0, np.nan))

def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series) -> pd.Series:
    """
    Money Flow Index (MFI) - "RSI cua volume".
    MFI > 80 = overbought, MFI < 20 = oversold.
    Khac RSI: co tinh volume vao, nen nang hon.
    """
    p = CFG.RSI_PERIOD
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    tp_diff = typical_price.diff()

    pos_flow = money_flow.where(tp_diff > 0, 0)
    neg_flow = money_flow.where(tp_diff < 0, 0)

    pos_sum = pos_flow.rolling(window=p, min_periods=p).sum()
    neg_sum = neg_flow.rolling(window=p, min_periods=p).sum()

    mfi_ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfi_ratio))

def _ad_line(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series) -> pd.Series:
    """
    Accumulation/Distribution Line.
    CLV = [(close - low) - (high - close)] / (high - low)
    Smart money tich luy (mua) hay phan phoi (ban)?
    """
    hl_range = high - low
    hl_range = hl_range.replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range
    return (clv * volume).cumsum()

def _cmf(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series) -> pd.Series:
    """
    Chaikin Money Flow (CMF) - A/D line normalized.
    CMF > 0 = ap luc mua, CMF < 0 = ap luc ban.
    Window = BB_PERIOD (100 phut).
    """
    w = CFG.BB_PERIOD
    hl_range = high - low
    hl_range = hl_range.replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range
    mf_volume = clv * volume

    cmf = mf_volume.rolling(w, min_periods=w).sum() / volume.rolling(w, min_periods=w).sum().replace(0, np.nan)
    return cmf

def _rvol(volume: pd.Series) -> pd.Series:
    """
    Relative Volume (RVOL) = volume hien tai / trung binh 60 phut.
    RVOL > 2 = volume bat thuong cao (co the co tin lon).
    RVOL < 0.5 = thi truong im ang.
    """
    w = CFG.VOL_LONG_WINDOW  # 60 phut
    avg_vol = volume.rolling(window=w, min_periods=w).mean()
    return volume / avg_vol.replace(0, np.nan)

def _volume_price_corr(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Volume-Price Correlation over BB_PERIOD.
    Corr > 0 = volume tang khi gia tang (bullish confirmation).
    Corr < 0 = volume tang khi gia giam (bearish pressure).
    Corr ~0 = gia va volume khong lien quan.
    """
    w = CFG.BB_PERIOD
    price_change = close.pct_change()
    return price_change.rolling(w, min_periods=w).corr(volume.pct_change())

def _volume_price_divergence(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Volume-Price Divergence: gia tang nhung volume giam = weak move.
    Tinh: sign(price_change) * sign(volume_change).
    -1 = divergence (canh bao dao chieu).
    +1 = confirmation (xu huong manh).
    Smooth bang EMA de giam nhieu.
    """
    w = CFG.VOL_SHORT_WINDOW
    price_dir = np.sign(close.diff(w))
    vol_dir = np.sign(volume.diff(w))
    raw = price_dir * vol_dir
    # Smooth bang rolling mean de co gia tri lien tuc
    return raw.rolling(w, min_periods=1).mean()

def _volume_momentum(volume: pd.Series) -> pd.Series:
    """
    Volume momentum = toc do thay doi volume.
    Volume dang tang toc hay giam toc?
    """
    w = CFG.VOL_SHORT_WINDOW
    vol_ma_short = volume.rolling(w, min_periods=w).mean()
    vol_ma_long = volume.rolling(CFG.VOL_LONG_WINDOW, min_periods=CFG.VOL_LONG_WINDOW).mean()
    return (vol_ma_short / vol_ma_long.replace(0, np.nan)) - 1


# ============================================================
# 3. TECHNICAL INDICATORS
# ============================================================

def _rsi(close: pd.Series) -> pd.Series:
    """RSI - Wilder's smoothing"""
    p = CFG.RSI_PERIOD
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / p, min_periods=p, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / p, min_periods=p, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def _macd_histogram(close: pd.Series) -> pd.Series:
    """MACD Histogram"""
    f, s, sig = CFG.MACD_FAST, CFG.MACD_SLOW, CFG.MACD_SIGNAL
    ema_f = close.ewm(span=f, min_periods=f, adjust=False).mean()
    ema_s = close.ewm(span=s, min_periods=s, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, min_periods=sig, adjust=False).mean()
    return macd - signal

def _bollinger_band_width(close: pd.Series) -> pd.Series:
    """BB Width"""
    p, n = CFG.BB_PERIOD, CFG.BB_STD
    sma = close.rolling(window=p, min_periods=p).mean()
    std = close.rolling(window=p, min_periods=p).std()
    upper = sma + n * std
    lower = sma - n * std
    return (upper - lower) / sma.replace(0, np.nan)

def _bb_position(close: pd.Series) -> pd.Series:
    """
    BB %B - vi tri gia trong Bollinger Band.
    0 = sat lower band, 1 = sat upper band, 0.5 = giua.
    > 1 = vuot upper, < 0 = vuot lower.
    """
    p, n = CFG.BB_PERIOD, CFG.BB_STD
    sma = close.rolling(window=p, min_periods=p).mean()
    std = close.rolling(window=p, min_periods=p).std()
    upper = sma + n * std
    lower = sma - n * std
    band_range = upper - lower
    band_range = band_range.replace(0, np.nan)
    return (close - lower) / band_range


# ============================================================
# 4. CANDLE PATTERN FEATURES
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
    """Tao tat ca features tu OHLCV. In-place."""
    rows_before = len(df)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    log_ret = _log_return(close)

    # === 1. PRICE ===
    df['log_return_1m'] = log_ret
    df['return_5m'] = _return_short(close)
    df['return_15m'] = _return_long(close)
    df['rolling_vol_15m'] = _rolling_vol_short(log_ret)
    df['rolling_vol_60m'] = _rolling_vol_long(log_ret)
    del log_ret

    # === 2. VOLUME (MOI - 9 features) ===
    df['volume_zscore_15m'] = _volume_zscore(volume)
    df['volume_change'] = _volume_change(volume)
    df['obv_slope'] = _obv_slope(close, volume)
    df['mfi'] = _mfi(high, low, close, volume)
    df['cmf'] = _cmf(high, low, close, volume)
    df['rvol'] = _rvol(volume)
    df['vol_price_corr'] = _volume_price_corr(close, volume)
    df['vol_price_div'] = _volume_price_divergence(close, volume)
    df['vol_momentum'] = _volume_momentum(volume)

    # === 3. TECHNICAL ===
    df['rsi_14'] = _rsi(close)
    df['macd_histogram'] = _macd_histogram(close)
    df['bb_width'] = _bollinger_band_width(close)
    df['bb_position'] = _bb_position(close)

    # === 4. CANDLE ===
    o = df['open'].values
    h = high.values
    l = low.values
    c = close.values
    df['candle_body_ratio'] = _candle_body_ratio(o, h, l, c)
    df['upper_wick_ratio'] = _upper_wick_ratio(o, h, l, c)
    df['lower_wick_ratio'] = _lower_wick_ratio(o, h, l, c)
    del o, h, l, c

    # === 5. TIME FEATURES (chieu du lieu hoan toan moi) ===
    # Sin/cos encoding de model hieu "23h gan 0h" (cyclical)
    hour = df.index.hour
    dow = df.index.dayofweek  # 0=Mon, 6=Sun

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7).astype(np.float32)

    # Session flags: phien My co volume cao va trend manh hon
    df['is_us_session'] = ((hour >= 13) & (hour <= 21)).astype(np.float32)
    df['is_asia_session'] = ((hour >= 1) & (hour <= 8)).astype(np.float32)

    # Drop NaN
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
        # Price (5)
        'log_return_1m',
        'return_5m',
        'return_15m',
        'rolling_vol_15m',
        'rolling_vol_60m',
        # Volume (9)
        'volume_zscore_15m',
        'volume_change',
        'obv_slope',
        'mfi',
        'cmf',
        'rvol',
        'vol_price_corr',
        'vol_price_div',
        'vol_momentum',
        # Technical (4)
        'rsi_14',
        'macd_histogram',
        'bb_width',
        'bb_position',
        # Candle (3)
        'candle_body_ratio',
        'upper_wick_ratio',
        'lower_wick_ratio',
        # Time (6)
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos',
        'is_us_session',
        'is_asia_session',
    ]
