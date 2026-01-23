"""检查价格前复权转换是否正确"""
import pandas as pd
import sys
import io

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 读取K线数据
df = pd.read_csv('data/raw/kline/kline_all.csv', dtype={'stock_code': str})

# 检查000882.SZ
df_000882 = df[df['instrument'] == '000882.SZ'].copy()
df_000882['date'] = pd.to_datetime(df_000882['date'])
df_000882 = df_000882.sort_values('date')

# 获取最新的复权因子
latest_factor = df_000882['adjust_factor'].iloc[-1]
print(f"000882.SZ 最新的复权因子: {latest_factor}")

# 获取2025-01-02的数据
df_20250102 = df_000882[df_000882['date'] == '2025-01-02']
if not df_20250102.empty:
    close_hfq = df_20250102['close'].iloc[0]
    adjust_factor_20250102 = df_20250102['adjust_factor'].iloc[0]
    close_qfq = close_hfq / latest_factor
    print(f"\n2025-01-02 的数据:")
    print(f"  - 收盘价（后复权）: {close_hfq:.6f}")
    print(f"  - 当日复权因子: {adjust_factor_20250102:.6f}")
    print(f"  - 最新复权因子: {latest_factor:.6f}")
    print(f"  - 收盘价（前复权，代码计算）: {close_qfq:.2f}")
    print(f"  - 转换公式: {close_hfq:.6f} / {latest_factor:.6f} = {close_qfq:.2f}")
    print(f"\n用户看到的东方财富前复权价格: 2.29元")
    print(f"代码计算的前复权价格: {close_qfq:.2f}元")
    print(f"差异: {abs(close_qfq - 2.29):.2f}元")

# 检查603068.SH
print("\n" + "="*60)
df_603068 = df[df['instrument'] == '603068.SH'].copy()
df_603068['date'] = pd.to_datetime(df_603068['date'])
df_603068 = df_603068.sort_values('date')

latest_factor_603068 = df_603068['adjust_factor'].iloc[-1]
print(f"603068.SH 最新的复权因子: {latest_factor_603068}")

# 获取2024-11-13的数据
df_20241113 = df_603068[df_603068['date'] == '2024-11-13']
if not df_20241113.empty:
    close_hfq = df_20241113['close'].iloc[0]
    adjust_factor_20241113 = df_20241113['adjust_factor'].iloc[0]
    close_qfq = close_hfq / latest_factor_603068
    print(f"\n2024-11-13 的数据:")
    print(f"  - 收盘价（后复权）: {close_hfq:.6f}")
    print(f"  - 当日复权因子: {adjust_factor_20241113:.6f}")
    print(f"  - 最新复权因子: {latest_factor_603068:.6f}")
    print(f"  - 收盘价（前复权，代码计算）: {close_qfq:.2f}")
    print(f"\n用户看到的东方财富前复权价格（2024-11-13）: 32.04元")
    print(f"代码计算的前复权价格: {close_qfq:.2f}元")
    print(f"差异: {abs(close_qfq - 32.04):.2f}元")

# 检查002397.SZ
print("\n" + "="*60)
df_002397 = df[df['instrument'] == '002397.SZ'].copy()
df_002397['date'] = pd.to_datetime(df_002397['date'])
df_002397 = df_002397.sort_values('date')

latest_factor_002397 = df_002397['adjust_factor'].iloc[-1]
print(f"002397.SZ 最新的复权因子: {latest_factor_002397}")

# 获取2025-01-02的数据
df_002397_20250102 = df_002397[df_002397['date'] == '2025-01-02']
if not df_002397_20250102.empty:
    close_hfq = df_002397_20250102['close'].iloc[0]
    adjust_factor_20250102 = df_002397_20250102['adjust_factor'].iloc[0]
    close_qfq = close_hfq / latest_factor_002397
    print(f"\n2025-01-02 的数据:")
    print(f"  - 收盘价（后复权）: {close_hfq:.6f}")
    print(f"  - 当日复权因子: {adjust_factor_20250102:.6f}")
    print(f"  - 最新复权因子: {latest_factor_002397:.6f}")
    print(f"  - 收盘价（前复权，代码计算）: {close_qfq:.2f}")
    print(f"\n用户看到的东方财富前复权价格（2025-01-02）: 3.23元")
    print(f"代码计算的前复权价格: {close_qfq:.2f}元")
    print(f"差异: {abs(close_qfq - 3.23):.2f}元")
