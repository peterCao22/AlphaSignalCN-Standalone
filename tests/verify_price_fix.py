"""验证价格前复权修复效果"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from scripts.kline_processing import process_kline

print("="*80)
print("测试1: 验证 process_kline 不会覆盖已存在的前复权价格")
print("="*80)

# 读取K线数据
df = pd.read_csv('data/raw/kline/kline_all.csv', dtype={'stock_code': str})

# 测试 000882.SZ
df_000882 = df[df['instrument'] == '000882.SZ'].copy()
df_000882['date'] = pd.to_datetime(df_000882['date'])
df_000882 = df_000882.sort_values('date')

# 获取2025-01-02的数据
df_test = df_000882[df_000882['date'] <= '2025-01-02'].tail(20).copy()

# 手动进行前复权转换（模拟 _load_local_kline_data 的行为）
latest_factor = df_test['adjust_factor'].iloc[-1]
for col in ['open', 'high', 'low', 'close']:
    if col in df_test.columns:
        df_test[f'{col}_qfq'] = df_test[col] / latest_factor

print(f"\n000882.SZ 最新复权因子: {latest_factor}")
print(f"\n前复权转换前（2025-01-02）:")
row_before = df_test[df_test['date'] == '2025-01-02'].iloc[0]
print(f"  - close（后复权）: {row_before['close']:.6f}")
print(f"  - close_qfq（前复权，手动转换）: {row_before['close_qfq']:.6f}")

# 调用 process_kline（应该不会覆盖 close_qfq）
df_processed = process_kline(df_test)

print(f"\n调用 process_kline 后（2025-01-02）:")
row_after = df_processed[df_processed['date'] == '2025-01-02'].iloc[0]
print(f"  - close（后复权）: {row_after['close']:.6f}")
print(f"  - close_qfq（前复权，process_kline 后）: {row_after['close_qfq']:.6f}")

# 验证价格是否被保留
if abs(row_after['close_qfq'] - row_before['close_qfq']) < 0.01:
    print(f"\n✅ 修复成功！前复权价格被正确保留: {row_after['close_qfq']:.2f}元")
else:
    print(f"\n❌ 修复失败！前复权价格被覆盖:")
    print(f"   期望: {row_before['close_qfq']:.2f}元")
    print(f"   实际: {row_after['close_qfq']:.2f}元")

print("\n" + "="*80)
print("测试2: 验证东方财富前复权价格一致性")
print("="*80)

expected_prices = {
    '000882.SZ': {'date': '2025-01-02', 'expected': 2.29},
    '603068.SH': {'date': '2024-11-13', 'expected': 32.04},
    '002397.SZ': {'date': '2025-01-02', 'expected': 3.23}
}

for symbol, info in expected_prices.items():
    df_stock = df[df['instrument'] == symbol].copy()
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_stock = df_stock.sort_values('date')
    
    # 手动前复权转换
    latest_factor = df_stock['adjust_factor'].iloc[-1]
    df_test = df_stock[df_stock['date'] <= info['date']].tail(20).copy()
    for col in ['open', 'high', 'low', 'close']:
        if col in df_test.columns:
            df_test[f'{col}_qfq'] = df_test[col] / latest_factor
    
    # 调用 process_kline
    df_processed = process_kline(df_test)
    
    # 获取指定日期的价格
    row = df_processed[df_processed['date'] == info['date']]
    if not row.empty:
        actual_price = row.iloc[0]['close_qfq']
        expected_price = info['expected']
        diff = abs(actual_price - expected_price)
        
        if diff < 0.01:
            print(f"✅ {symbol} {info['date']}: {actual_price:.2f}元（期望: {expected_price:.2f}元，差异: {diff:.4f}元）")
        else:
            print(f"❌ {symbol} {info['date']}: {actual_price:.2f}元（期望: {expected_price:.2f}元，差异: {diff:.4f}元）")

print("\n" + "="*80)
print("测试完成")
print("="*80)
