from multiprocessing import Pool
import pandas as pd
import pyarrow.parquet as pq
from glob import glob
import os
from tqdm import tqdm
import pytz
from datetime import datetime
import time
import gc

SOURCE_DIR = r"E:\tickdata_binance\databinance"
OUTPUT_DIR = r"E:\tickdata_binance\resample\5m"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_existing_files():
    """Lấy danh sách file đã resample xong"""
    existing_files = set()
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith("_5m.parquet"):
                symbol = file.replace("_5m.parquet", "")
                existing_files.add(symbol)
    return existing_files

def find_tickdata_files():
    """Tìm tất cả file tickdata parquet trong thư mục, loại bỏ file đã xử lý"""
    files = []
    existing_symbols = get_existing_files()
    
    print(f"🔍 Tìm file tickdata...")
    print(f"📁 Đã có {len(existing_symbols)} symbols được resample")
    
    for root, _, _ in os.walk(SOURCE_DIR):
        found_files = glob(os.path.join(root, "*-combined-aggtrades-*.parquet"))
        for file_path in found_files:
            symbol = os.path.basename(file_path).split("-")[0]
            if symbol not in existing_symbols:
                files.append(file_path)
            else:
                print(f"⏭️  Skip {symbol} (đã có)")
    
    return files

def resample_file_simple(file_path):
    """Resample file đơn giản và hiệu quả"""
    try:
        symbol = os.path.basename(file_path).split("-")[0]
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        
        print(f"Processing {symbol} ({file_size_gb:.1f}GB)...")
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu xử lý {symbol}")
        
        # Đọc file với PyArrow - chỉ đọc cột cần thiết
        parquet_file = pq.ParquetFile(file_path)
        
        # Tìm cột thời gian
        time_col = None
        for col in ["transact_time", "timestamp", "T"]:
            if col in parquet_file.schema.names:
                time_col = col
                break
        
        if time_col is None:
            return f"❌ {file_path}: Không tìm thấy cột thời gian"
        
        # Chọn cột cần thiết
        columns_to_read = ["price", time_col]
        volume_col = None
        for col in ["quantity", "qty", "volume"]:
            if col in parquet_file.schema.names:
                volume_col = col
                columns_to_read.append(col)
                break
        
        print(f"   ⏰ Time: {time_col}, Volume: {volume_col}")
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Đã xác định cột dữ liệu cho {symbol}")
        
        # Đọc file theo chunks để tiết kiệm memory
        ohlcv_chunks = []
        
        # Đọc metadata để biết số row groups
        metadata = parquet_file.metadata
        num_row_groups = metadata.num_row_groups
        
        print(f"   📖 Đọc {num_row_groups} row groups...")
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu đọc {num_row_groups} row groups cho {symbol}")
        
        for rg_idx in range(num_row_groups):
            print(f"   📖 Row group {rg_idx + 1}/{num_row_groups}...")
            
            # Đọc row group
            table = parquet_file.read_row_group(rg_idx, columns=columns_to_read)
            df_chunk = table.to_pandas()
            
            if df_chunk.empty:
                continue
            
            # Chuyển đổi timestamp
            df_chunk['timestamp'] = pd.to_datetime(df_chunk[time_col], unit='ms', utc=True)
            df_chunk.set_index('timestamp', inplace=True)
            df_chunk.sort_index(inplace=True)
            
            # Resample chunk - ĐÂY LÀ VỊ TRÍ CHUYỂN ĐỔI TICKDATA THÀNH NẾN
            print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu resample row group {rg_idx + 1} thành nến 1m cho {symbol}")
            ohlc = df_chunk['price'].resample('5min').ohlc().dropna()
            print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hoàn thành resample row group {rg_idx + 1}: {len(ohlc)} nến cho {symbol}")
            
            # Volume nếu có
            if volume_col and volume_col in df_chunk.columns:
                volume = df_chunk[volume_col].resample('5min').sum().dropna()
                ohlc['volume'] = volume
            
            ohlcv_chunks.append(ohlc)
            
            # Giải phóng memory
            del df_chunk, table
            gc.collect()
        
        # Kết hợp tất cả chunks
        print(f"    Combining {len(ohlcv_chunks)} chunks...")
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu kết hợp {len(ohlcv_chunks)} chunks cho {symbol}")
        
        if not ohlcv_chunks:
            return f"❌ {file_path}: Không có dữ liệu hợp lệ"
        
        ohlcv_df = pd.concat(ohlcv_chunks)
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hoàn thành kết hợp chunks: {len(ohlcv_df)} nến cho {symbol}")
        
        # Gộp các candles cùng thời gian
        ohlcv_df = ohlcv_df.groupby(ohlcv_df.index).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum' if 'volume' in ohlcv_df.columns else 'first'
        })
        
        # Đảm bảo index có timezone UTC
        if ohlcv_df.index.tz is None:
            ohlcv_df.index = ohlcv_df.index.tz_localize('UTC')
        
        # Lưu file
        out_path = os.path.join(OUTPUT_DIR, f"{symbol}_5m.parquet")
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu lưu file {symbol}_5m.parquet")
        ohlcv_df.to_parquet(out_path, engine='pyarrow')
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hoàn thành lưu file {symbol}_5m.parquet")
        
        # Lưu số candles trước khi xóa biến
        num_candles = len(ohlcv_df)
        
        print(f"   💾 Đã lưu: {out_path}")
        print(f"   📈 {num_candles:,} candles tạo được")
        
        # Giải phóng memory
        del ohlcv_chunks, ohlcv_df
        gc.collect()
        
        return f"✅ {symbol}: {num_candles:,} candles ({file_size_gb:.1f}GB)"
        
    except Exception as e:
        return f"❌ Error with {file_path}: {str(e)}"

def resample_1m(file_path):
    """Resample tickdata thành nến 1 phút - phiên bản đơn giản"""
    try:
        symbol = os.path.basename(file_path).split("-")[0]
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        
        # Nếu file lớn (>2GB), sử dụng chunking
        if file_size_gb > 2:
            return resample_file_simple(file_path)
        
        # File nhỏ, xử lý trực tiếp
        print(f"Processing {symbol} ({file_size_gb:.1f}GB)...")
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu xử lý file nhỏ {symbol}")
        
        # Đọc file với PyArrow
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        # Tìm cột thời gian
        time_col = None
        for col in ["transact_time", "timestamp", "T"]:
            if col in df.columns:
                time_col = col
                break
        
        if time_col is None:
            return f"❌ {file_path}: Không tìm thấy cột thời gian"
        
        # Chuyển đổi timestamp
        df['timestamp'] = pd.to_datetime(df[time_col], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Resample thành OHLCV - ĐÂY LÀ VỊ TRÍ CHUYỂN ĐỔI TICKDATA THÀNH NẾN
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu resample tickdata thành nến 1m cho {symbol}")
        ohlc = df['price'].resample('5min').ohlc().dropna()
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hoàn thành resample: {len(ohlc)} nến cho {symbol}")
        
        # Volume nếu có
        volume_col = None
        for col in ["quantity", "qty", "volume"]:
            if col in df.columns:
                volume_col = col
                break
        
        if volume_col:
            volume = df[volume_col].resample('5min').sum().dropna()
            ohlc['volume'] = volume
        
        # Đảm bảo index có timezone UTC
        if ohlc.index.tz is None:
            ohlc.index = ohlc.index.tz_localize('UTC')
        
        # Lưu file
        out_path = os.path.join(OUTPUT_DIR, f"{symbol}_5m.parquet")
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu lưu file {symbol}_5m.parquet")
        ohlc.to_parquet(out_path, engine='pyarrow')
        print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hoàn thành lưu file {symbol}_5m.parquet")
        
        return f"✅ {symbol}: {len(ohlc):,} candles ({file_size_gb:.1f}GB)"
        
    except Exception as e:
        return f"❌ Error with {file_path}: {str(e)}"

def run_parallel(files, n_proc=2):  # Giảm số process để tránh quá tải memory
    """Chạy resample song song với progress chi tiết"""
    results = []
    
    print(f"Bắt đầu resample {len(files)} files với {n_proc} processes...")
    print("💡 Sử dụng PyArrow chunking cho file lớn")
    print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu quá trình resample song song")
    
    with Pool(processes=n_proc) as pool:
        with tqdm(total=len(files), desc="Resampling", unit="file") as pbar:
            for result in pool.imap_unordered(resample_1m, files):
                results.append(result)
                pbar.update()
                
                # Hiển thị kết quả ngay lập tức
                if result.startswith("✅"):
                    pbar.write(f"  {result}")
                elif result.startswith("❌"):
                    pbar.write(f"  {result}")
    
    print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Hoàn thành quá trình resample song song")
    return results

def analyze_results(results):
    """Phân tích kết quả resample"""
    success_count = sum(1 for r in results if r.startswith("✅"))
    error_count = sum(1 for r in results if r.startswith("❌"))
    
    print(f"\n📊 Kết quả resample:")
    print(f"✅ Thành công: {success_count}")
    print(f"❌ Lỗi: {error_count}")
    print(f"📈 Tỷ lệ thành công: {success_count/(success_count+error_count)*100:.1f}%")
    
    # Hiển thị một số lỗi mẫu
    errors = [r for r in results if r.startswith("❌")]
    
    if errors:
        print(f"\nMột số lỗi mẫu:")
        for error in errors[:3]:
            print(f"  {error}")

def display_resampled_data(symbol, num_rows=10):
    """Hiển thị dữ liệu từ file đã resample với timestamp"""
    try:
        file_path = os.path.join(OUTPUT_DIR, f"{symbol}_5m.parquet")
        
        if not os.path.exists(file_path):
            print(f"❌ File {file_path} không tồn tại")
            return
        
        # Đọc file parquet
        df = pd.read_parquet(file_path)
        
        print(f"\n�� Dữ liệu nến 1m của {symbol}:")
        print(f"📁 File: {file_path}")
        print(f"📈 Tổng số nến: {len(df):,}")
        print(f"⏰ Thời gian từ: {df.index.min()} đến {df.index.max()}")
        
        # Hiển thị 10 dòng đầu tiên
        print(f"\n🔝 {num_rows} dòng đầu tiên:")
        print("=" * 80)
        print(f"{'Timestamp':<20} {'Open':<12} {'High':<12} {'Low':<12} {'Close':<12} {'Volume':<12}")
        print("-" * 80)
        
        for i, (timestamp, row) in enumerate(df.head(num_rows).iterrows()):
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            print(f"{timestamp_str:<20} {row['open']:<12.2f} {row['high']:<12.2f} {row['low']:<12.2f} {row['close']:<12.2f} {row.get('volume', 0):<12.2f}")
        
        # Hiển thị 10 dòng cuối cùng
        print(f"\n�� {num_rows} dòng cuối cùng:")
        print("=" * 80)
        print(f"{'Timestamp':<20} {'Open':<12} {'High':<12} {'Low':<12} {'Close':<12} {'Volume':<12}")
        print("-" * 80)
        
        for i, (timestamp, row) in enumerate(df.tail(num_rows).iterrows()):
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            print(f"{timestamp_str:<20} {row['open']:<12.2f} {row['high']:<12.2f} {row['low']:<12.2f} {row['close']:<12.2f} {row.get('volume', 0):<12.2f}")
        
        # Thống kê cơ bản
        print(f"\n📈 Thống kê cơ bản:")
        print(f"   💰 Giá cao nhất: {df['high'].max():.2f}")
        print(f"   �� Giá thấp nhất: {df['low'].min():.2f}")
        print(f"   �� Volume trung bình: {df.get('volume', pd.Series([0])).mean():.2f}")
        print(f"   📊 Volume tổng: {df.get('volume', pd.Series([0])).sum():.2f}")
        
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {symbol}: {str(e)}")

def show_sample_resampled_files():
    """Hiển thị dữ liệu mẫu từ các file đã resample"""
    try:
        if not os.path.exists(OUTPUT_DIR):
            print(f"❌ Thư mục {OUTPUT_DIR} không tồn tại")
            return
        
        # Lấy danh sách file đã resample
        resampled_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith("_5m.parquet")]
        
        if not resampled_files:
            print(f"❌ Không có file nào đã được resample trong {OUTPUT_DIR}")
            return
        
        print(f"\n�� Hiển thị dữ liệu mẫu từ {len(resampled_files)} file đã resample:")
        
        # Hiển thị 3 file đầu tiên
        for i, file_name in enumerate(resampled_files[:3]):
            symbol = file_name.replace("_5m.parquet", "")
            display_resampled_data(symbol, num_rows=5)
            
            if i < 2:  # Không in dấu phân cách cho file cuối
                print("\n" + "="*100 + "\n")
        
        if len(resampled_files) > 3:
            print(f"\n... và {len(resampled_files) - 3} file khác")
            
    except Exception as e:
        print(f"❌ Lỗi khi hiển thị dữ liệu mẫu: {str(e)}")

if __name__ == "__main__":
    print("🚀 Bắt đầu resample tickdata thành nến 1 phút...")
    print("💪 Sử dụng PyArrow chunking cho hiệu suất cao")
    print(f"📁 Thư mục nguồn: {SOURCE_DIR}")
    print(f"📁 Thư mục đích: {OUTPUT_DIR}")
    print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Khởi động chương trình resample")
    
    # Tìm files (skip những file đã có)
    files = find_tickdata_files()
    print(f"📦 Cần resample {len(files)} files")
    
    if not files:
        print("✅ Tất cả files đã được resample!")
        exit(0)
    
    # Hiển thị thông tin files
    print("\n📋 Thông tin files cần xử lý:")
    for file_path in files[:5]:  # Hiển thị 5 file đầu
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        symbol = os.path.basename(file_path).split("-")[0]
        print(f"   {symbol}: {file_size_gb:.1f}GB")
    
    if len(files) > 5:
        print(f"   ... và {len(files) - 5} files khác")
    
    # Chạy resample
    start_time = time.time()
    print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu quá trình resample chính")
    results = run_parallel(files, n_proc=2)  # Giảm số process để tránh quá tải memory
    end_time = time.time()
    print(f"⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Kết thúc quá trình resample chính")
    
    # Hiển thị dữ liệu mẫu sau khi resample
    print(f"\n🎉 Hoàn thành resample! Hiển thị dữ liệu mẫu:")
    show_sample_resampled_files()

    # Lưu log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"resample_log_{timestamp}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Resample log - {datetime.now()}\n")
        f.write(f"Thời gian chạy: {end_time - start_time:.1f}s\n")
        f.write("Sử dụng PyArrow chunking\n")
        f.write("="*50 + "\n")
        f.write("\n".join(results))
    
    # Phân tích kết quả
    analyze_results(results)
    
    print(f"✅ Hoàn thành! Log đã lưu vào {log_file}")
    print(f"⏱️  Tổng thời gian: {end_time - start_time:.1f}s")