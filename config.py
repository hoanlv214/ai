"""
config.py - Cau hinh tap trung cho toan bo pipeline

HUONG DAN:
- Chay binh thuong: sua config o day, chay `python main.py`
- Chay Optuna optimizer: chay `python optimizer.py`
  (Optuna se tu dong update config cho tung trial)

Khi doi BAR_MINUTES: goi CFG.update_timeframe(bar, predict) de tu dong
tinh lai tat ca window sizes.
"""

import os


class CFG:

    # ============================================================
    # DATA FILES - Mapping timeframe (phut) -> file path
    # ============================================================
    DATA_DIR = r"E:\tickdata_binance\resample"
    DATA_FILES = {
        1: os.path.join(DATA_DIR, "1m", "BTCUSDT_1m.parquet"),
        5: os.path.join(DATA_DIR, "5m", "BTCUSDT_5m.parquet"),
    }

    # ============================================================
    # TIMEFRAME HIEN TAI (duoc update boi optimizer hoac thu cong)
    # ============================================================
    BAR_MINUTES = 5             # Nen hien tai (1, 5, 15, 30, 60)
    PREDICT_MINUTES = 15        # Du doan bao nhieu phut toi
    DATA_PATH = DATA_FILES.get(5, "")

    # Tu dong tinh
    PREDICT_BARS = PREDICT_MINUTES // BAR_MINUTES  # = 3

    USE_FLOAT32 = True

    # ============================================================
    # FEATURE WINDOWS - Dinh nghia bang PHUT (tuyet doi)
    # Khi doi BAR_MINUTES, se tu dong convert sang so bars
    # ============================================================
    FEAT_RETURN_SHORT_MIN = 5       # return ngan (5 phut)
    FEAT_RETURN_LONG_MIN = 15       # return dai (15 phut)
    FEAT_VOL_SHORT_MIN = 15         # vol ngan (15 phut)
    FEAT_VOL_LONG_MIN = 60          # vol dai (60 phut)
    FEAT_ZSCORE_MIN = 15            # volume zscore (15 phut)

    # Auto-computed (bars) - se duoc update boi update_timeframe()
    RETURN_SHORT_WINDOW = max(1, FEAT_RETURN_SHORT_MIN // BAR_MINUTES)   # 1
    RETURN_LONG_WINDOW = max(1, FEAT_RETURN_LONG_MIN // BAR_MINUTES)     # 3
    VOL_SHORT_WINDOW = max(1, FEAT_VOL_SHORT_MIN // BAR_MINUTES)         # 3
    VOL_LONG_WINDOW = max(1, FEAT_VOL_LONG_MIN // BAR_MINUTES)           # 12
    VOLUME_ZSCORE_WINDOW = max(1, FEAT_ZSCORE_MIN // BAR_MINUTES)        # 3

    # Technical Indicators - Dinh nghia bang PHUT (tuyet doi)
    # Dam bao so sanh 1m vs 5m cong bang: cung lookback time
    # VD: RSI nhin 70 phut = RSI(70) tren 1m = RSI(14) tren 5m
    FEAT_RSI_MIN = 70             # RSI lookback 70 phut
    FEAT_MACD_FAST_MIN = 60       # MACD fast 60 phut
    FEAT_MACD_SLOW_MIN = 130      # MACD slow 130 phut
    FEAT_MACD_SIGNAL_MIN = 45     # MACD signal 45 phut
    FEAT_BB_MIN = 100             # Bollinger Band 100 phut
    BB_STD = 2.0

    # Auto-computed (bars) - update boi update_timeframe()
    RSI_PERIOD = max(1, FEAT_RSI_MIN // BAR_MINUTES)           # 14
    MACD_FAST = max(1, FEAT_MACD_FAST_MIN // BAR_MINUTES)      # 12
    MACD_SLOW = max(1, FEAT_MACD_SLOW_MIN // BAR_MINUTES)      # 26
    MACD_SIGNAL = max(1, FEAT_MACD_SIGNAL_MIN // BAR_MINUTES)  # 9
    BB_PERIOD = max(1, FEAT_BB_MIN // BAR_MINUTES)             # 20

    # ============================================================
    # LABELING
    # ============================================================
    LABEL_THRESHOLD = 0.0

    # ============================================================
    # TRAIN/TEST SPLIT
    # ============================================================
    TRAIN_END_DATE = None
    TEST_START_DATE = None
    TRAIN_RATIO = 0.8

    # ============================================================
    # MODEL - LightGBM (co the duoc Optuna tune)
    # ============================================================
    LGB_N_ESTIMATORS = 500
    LGB_MAX_DEPTH = 6
    LGB_LEARNING_RATE = 0.05
    LGB_NUM_LEAVES = 31
    LGB_MIN_CHILD_SAMPLES = 50
    LGB_SUBSAMPLE = 0.8
    LGB_COLSAMPLE = 0.8
    LGB_IS_UNBALANCE = True
    RANDOM_STATE = 42

    # ============================================================
    # BACKTEST & EVALUATE
    # ============================================================
    PROB_THRESHOLD = 0.6
    PROB_BINS = [
        (0.00, 0.40),
        (0.40, 0.45),
        (0.45, 0.50),
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 1.01),
    ]

    # ============================================================
    # OPTIMIZER (Optuna)
    # ============================================================
    # Cac timeframe data nen de test (phai co file hoac resample duoc)
    CANDIDATE_DATA_TF = [1, 5]
    # Cac timeframe predict de test (phut)
    CANDIDATE_PREDICT_TF = [5, 15, 30, 60]
    # So trials Optuna
    OPTUNA_N_TRIALS = 50
    # Metric toi uu: 'auc', 'sharpe', 'winrate', 'f1'
    OPTUNA_METRIC = 'auc'
    # Tune LGB hyperparams cung timeframe?
    OPTUNA_TUNE_LGB = True

    # ============================================================
    # METHODS
    # ============================================================

    @classmethod
    def update_timeframe(cls, bar_minutes: int, predict_minutes: int):
        """
        Cap nhat config cho 1 combo timeframe cu the.
        Tu dong tinh lai tat ca window sizes.

        Parameters
        ----------
        bar_minutes : int
            Nen bao nhieu phut (1, 5, 15, 30, 60)
        predict_minutes : int
            Du doan bao nhieu phut toi (5, 15, 30, 60, 120, 240)
        """
        cls.BAR_MINUTES = bar_minutes
        cls.PREDICT_MINUTES = predict_minutes
        cls.PREDICT_BARS = predict_minutes // bar_minutes

        # Data path
        if bar_minutes in cls.DATA_FILES:
            cls.DATA_PATH = cls.DATA_FILES[bar_minutes]

        # Feature windows: convert phut -> bars
        cls.RETURN_SHORT_WINDOW = max(1, cls.FEAT_RETURN_SHORT_MIN // bar_minutes)
        cls.RETURN_LONG_WINDOW = max(1, cls.FEAT_RETURN_LONG_MIN // bar_minutes)
        cls.VOL_SHORT_WINDOW = max(1, cls.FEAT_VOL_SHORT_MIN // bar_minutes)
        cls.VOL_LONG_WINDOW = max(1, cls.FEAT_VOL_LONG_MIN // bar_minutes)
        cls.VOLUME_ZSCORE_WINDOW = max(1, cls.FEAT_ZSCORE_MIN // bar_minutes)

        # Indicator periods: convert phut -> bars
        cls.RSI_PERIOD = max(2, cls.FEAT_RSI_MIN // bar_minutes)
        cls.MACD_FAST = max(2, cls.FEAT_MACD_FAST_MIN // bar_minutes)
        cls.MACD_SLOW = max(2, cls.FEAT_MACD_SLOW_MIN // bar_minutes)
        cls.MACD_SIGNAL = max(2, cls.FEAT_MACD_SIGNAL_MIN // bar_minutes)
        cls.BB_PERIOD = max(2, cls.FEAT_BB_MIN // bar_minutes)

    @classmethod
    def update_lgb_params(cls, **params):
        """Cap nhat LightGBM params (dung boi Optuna)."""
        mapping = {
            'n_estimators': 'LGB_N_ESTIMATORS',
            'max_depth': 'LGB_MAX_DEPTH',
            'learning_rate': 'LGB_LEARNING_RATE',
            'num_leaves': 'LGB_NUM_LEAVES',
            'min_child_samples': 'LGB_MIN_CHILD_SAMPLES',
            'subsample': 'LGB_SUBSAMPLE',
            'colsample_bytree': 'LGB_COLSAMPLE',
        }
        for key, val in params.items():
            attr = mapping.get(key, key)
            if hasattr(cls, attr):
                setattr(cls, attr, val)

    @classmethod
    def get_combo_name(cls) -> str:
        """Tra ve ten combo hien tai, vd: '5m_15m'"""
        return f"{cls.BAR_MINUTES}m_{cls.PREDICT_MINUTES}m"

    @classmethod
    def label_col(cls) -> str:
        """Ten cot label dong, vd: 'future_return_15m', 'future_return_60m'"""
        return f"future_return_{cls.PREDICT_MINUTES}m"

    @classmethod
    def print_config(cls):
        """In toan bo config."""
        print("=" * 60)
        print("  PIPELINE CONFIGURATION")
        print("=" * 60)
        print(f"  Combo:        {cls.get_combo_name()}")
        print(f"  Data:         {cls.DATA_PATH}")
        print(f"  Bar:          {cls.BAR_MINUTES}m -> Predict: {cls.PREDICT_MINUTES}m ({cls.PREDICT_BARS} bars)")
        print(f"  Float32:      {cls.USE_FLOAT32}")
        print(f"  Threshold:    {cls.LABEL_THRESHOLD}")
        print(f"  Split:        {cls.TRAIN_END_DATE or f'auto {cls.TRAIN_RATIO:.0%}'}")
        print(f"  Windows:      ret_short={cls.RETURN_SHORT_WINDOW}b({cls.FEAT_RETURN_SHORT_MIN}m) "
              f"ret_long={cls.RETURN_LONG_WINDOW}b({cls.FEAT_RETURN_LONG_MIN}m) "
              f"vol_short={cls.VOL_SHORT_WINDOW}b vol_long={cls.VOL_LONG_WINDOW}b")
        print(f"  RSI={cls.RSI_PERIOD}b({cls.FEAT_RSI_MIN}m) "
              f"MACD={cls.MACD_FAST}/{cls.MACD_SLOW}/{cls.MACD_SIGNAL}b "
              f"BB={cls.BB_PERIOD}b({cls.FEAT_BB_MIN}m, std={cls.BB_STD})")
        print(f"  LGB:          trees={cls.LGB_N_ESTIMATORS} depth={cls.LGB_MAX_DEPTH} "
              f"lr={cls.LGB_LEARNING_RATE} leaves={cls.LGB_NUM_LEAVES}")
        print(f"  Prob thresh:  {cls.PROB_THRESHOLD}")
        print("=" * 60)
