"""检查002397.SZ的开高低收数据"""
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

df = pd.read_csv('data/raw/kline/kline_all.csv')
df_002397 = df[df['instrument'] == '002397.SZ'].copy()
df_002397['date'] = pd.to_datetime(df_002397['date'])

# 获取2025-01-02的数据
row = df_002397[df_002397['date'] == '2025-01-02'].iloc[0]
latest_factor = df_002397['adjust_factor'].iloc[-1]

print("002397.SZ 2025-01-02 原始数据（后复权）:")
print(f"  open  = {row['open']:.6f}")
print(f"  high  = {row['high']:.6f}")
print(f"  low   = {row['low']:.6f}")
print(f"  close = {row['close']:.6f}")
print(f"  volume = {row['volume']:.0f}")
print(f"  adjust_factor = {row['adjust_factor']:.6f}")

print(f"\n最新复权因子: {latest_factor:.6f}")

print(f"\n前复权价格:")
open_qfq = row['open'] / latest_factor
high_qfq = row['high'] / latest_factor
low_qfq = row['low'] / latest_factor
close_qfq = row['close'] / latest_factor

print(f"  open_qfq  = {open_qfq:.4f}元")
print(f"  high_qfq  = {high_qfq:.4f}元")
print(f"  low_qfq   = {low_qfq:.4f}元")
print(f"  close_qfq = {close_qfq:.4f}元")

# 检查价格特征
print(f"\n价格特征:")
print(f"  开盘价 vs 收盘价差异: {abs(open_qfq - close_qfq) / close_qfq * 100:.4f}%")
print(f"  价格振幅（高-低）: {(high_qfq - low_qfq) / close_qfq * 100:.4f}%")

# 检查是否是完全一字板
is_perfect = abs(open_qfq - high_qfq) < 0.001 and abs(open_qfq - low_qfq) < 0.001 and abs(high_qfq - low_qfq) < 0.001
is_quasi = (high_qfq - low_qfq) / close_qfq < 0.005

print(f"\n一字板判断:")
print(f"  完全一字板（开=高=低=收）: {is_perfect}")
print(f"  准一字板（振幅<0.5%）: {is_quasi}")
