Tôi muốn xây dựng một hệ thống Machine Learning để:

* Nhận input là tickdata Binance (price, quantity, transact_time)
* Resample thành nến 1 phút
* Tính các technical indicators
* Tạo label dự đoán giá tăng/giảm sau 15 phút
* Train model trên dữ liệu lịch sử
* Test trên dữ liệu chưa train (time-series split)
* Output xác suất tăng giá (probability)

Yêu cầu:

---

# 1️⃣ YÊU CẦU CHUNG

* Sử dụng Python
* Sử dụng pandas, numpy
* Sử dụng LightGBM
* Code phải:

  * Chia module rõ ràng
  * Có docstring cho từng function
  * Có type hint
  * Comment chi tiết từng bước
  * Không được có data leakage
  * Không shuffle dữ liệu
  * Không dùng dữ liệu tương lai khi tính feature

---

# 2️⃣ CẤU TRÚC PROJECT YÊU CẦU

```id="g5w91m"
project/
│
├── data_loader.py
├── resampler.py
├── features.py
├── labeling.py
├── split.py
├── model.py
├── evaluate.py
├── backtest.py
└── main.py
```

---

# 3️⃣ CHI TIẾT MODULE

---

## data_loader.py

Yêu cầu:

* Đọc tickdata từ CSV
* Cột bao gồm:

  * price (float)
  * quantity (float)
  * transact_time (int, milliseconds)
* Convert transact_time sang datetime UTC
* Sort theo thời gian tăng dần
* Trả về pandas DataFrame index = datetime

---

## resampler.py

Yêu cầu:

Function:

```id="3oap6z"
def resample_to_1m(df: pd.DataFrame) -> pd.DataFrame
```

Thực hiện:

* Resample 1 minute
* Tính:

  * open
  * high
  * low
  * close
  * volume (sum quantity)
* Drop nến thiếu dữ liệu
* Không được forward fill giá

---

## features.py

Function chính:

```id="1d4xhy"
def create_features(df: pd.DataFrame) -> pd.DataFrame
```

Phải tạo các feature sau (chỉ dùng dữ liệu quá khứ):

### Price features

* log_return_1m
* return_5m
* return_15m
* rolling_vol_15m
* rolling_vol_60m

### Volume features

* volume_zscore_15m
* volume_change

### Technical indicators

* RSI(14)
* MACD histogram
* Bollinger band width
* candle_body_ratio
* upper_wick_ratio
* lower_wick_ratio

Tất cả indicator phải tự tính bằng pandas (không dùng thư viện ngoài như ta-lib).

Không dùng dữ liệu tương lai.

Drop NaN sau khi hoàn tất feature creation.

---

## labeling.py

Function:

```id="j9pt1v"
def create_label(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame
```

Tạo:

```id="4ot7yo"
future_return_15m = close.shift(-15) / close - 1
```

Label:

* Nếu threshold = 0:

  * y = 1 nếu future_return > 0
  * y = 0 nếu <= 0
* Nếu threshold > 0:

  * y = 1 nếu > threshold
  * y = 0 nếu < -threshold
  * drop vùng giữa

Drop các dòng cuối không đủ 15 phút.

---

## split.py

Function:

```id="oyy9mb"
def time_series_split(df, train_end_date, test_start_date)
```

Yêu cầu:

* Không shuffle
* Split theo thời gian
* Train: từ đầu đến train_end_date
* Test: từ test_start_date trở đi
* Không overlap

---

## model.py

Function:

```id="5v0v1q"
def train_model(X_train, y_train)
```

Yêu cầu:

* Sử dụng LightGBM classifier
* Không tuning phức tạp
* random_state cố định
* class_weight="balanced"
* Trả về model

Function:

```id="19m2b4"
def predict_proba(model, X_test)
```

Trả về xác suất tăng giá.

---

## evaluate.py

Phải tính:

* Accuracy
* Precision
* Recall
* F1
* AUC
* Confusion matrix

Thêm:

Function probability binning:

* Chia xác suất thành các bin:

  * 0.5–0.55
  * 0.55–0.6
  * 0.6–0.65
  * > 0.65
* In winrate từng bin

---

## backtest.py

Logic đơn giản:

```id="2a3vbb"
if prob > 0.6:
    position = 1
else:
    position = 0
```

Tính:

* strategy_return
* cumulative_return
* Sharpe ratio
* max drawdown

Không tính phí giao dịch ở version đầu.

---

## main.py

Phải:

1. Load tick data
2. Resample
3. Create feature
4. Create label
5. Split
6. Train
7. Predict
8. Evaluate
9. Backtest
10. In kết quả cuối cùng

---

# 4️⃣ YÊU CẦU QUAN TRỌNG

* Không được dùng dữ liệu tương lai để tính feature
* Không fit scaler trên toàn bộ data
* Không shuffle
* Không dùng cross validation random
* Code phải chạy được ngay nếu có CSV

---

# 5️⃣ OUTPUT MONG MUỐN

Cuối cùng in ra:

```id="z4ewt0"
Accuracy:
AUC:
Sharpe:
Max Drawdown:
Winrate khi prob > 0.6:
```

---

# 6️⃣ MỤC TIÊU CỦA PROJECT

Mục tiêu là kiểm tra xem:

"Chỉ từ tickdata resample thành nến và TA cơ bản, model có thể dự đoán xác suất giá tăng sau 15 phút hay không."

Không cần tối ưu hyperparameter ở version đầu.

---

# ✅ YÊU CẦU AI GENERATE CODE

* Viết đầy đủ code từng file
* Comment rõ ràng từng bước
* Đảm bảo không data leakage
* Code sạch, dễ đọc
* Không viết pseudo-code
* Viết production-ready

---