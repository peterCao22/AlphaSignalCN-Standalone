"""验证一字板识别修复效果"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path

# 模拟 LLMRerankService 的关键方法
def load_and_process_kline(symbol, trigger_date):
    """加载并处理K线数据"""
    from scripts.kline_processing import process_kline
    
    # 读取K线数据
    kline_path = Path('data/raw/kline/kline_all.csv')
    df_kline_all = pd.read_csv(kline_path, dtype={'stock_code': str})
    df_kline = df_kline_all[df_kline_all['instrument'] == symbol].copy()
    
    # 标准化日期格式
    df_kline['date'] = pd.to_datetime(df_kline['date'], format='ISO8601', errors='coerce')
    df_kline = df_kline.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    
    # 过滤到trigger_date之前
    trigger_dt = pd.to_datetime(trigger_date)
    df_kline = df_kline[df_kline['date'] <= trigger_dt].copy()
    
    # 只保留最近6个月
    cutoff_date = trigger_dt - pd.DateOffset(months=6)
    df_kline = df_kline[df_kline['date'] >= cutoff_date].copy()
    
    # 前复权转换
    if 'adjust_factor' in df_kline.columns:
        latest_factor = df_kline['adjust_factor'].iloc[-1]
        if pd.notna(latest_factor) and latest_factor > 0:
            for col in ['open', 'high', 'low', 'close']:
                if col in df_kline.columns:
                    df_kline[f'{col}_qfq'] = df_kline[col] / latest_factor
    
    # 处理K线数据（计算技术指标）
    df_processed = process_kline(df_kline)
    
    # 计算volume_ma20
    if 'volume' in df_processed.columns:
        valid_mask = df_processed['volume'].fillna(0) > 0
        df_processed['volume_ma20'] = np.nan
        if valid_mask.sum() > 0:
            df_valid = df_processed.loc[valid_mask].copy()
            df_valid['volume_ma20'] = df_valid['volume'].rolling(window=20).mean()
            df_processed.loc[df_valid.index, 'volume_ma20'] = df_valid['volume_ma20']
    
    return df_processed, trigger_dt

def identify_one_word_limit_up(df, symbol):
    """识别一字板"""
    # 获取涨停阈值
    stock_code = symbol.split('.')[0]
    if stock_code.startswith(('92', '43', '8')):
        limit_up_threshold = 0.30
    elif stock_code.startswith(('300', '301', '688', '689')):
        limit_up_threshold = 0.20
    else:
        limit_up_threshold = 0.10
    
    # 涨停判断
    if 'pct_change' not in df.columns or df['pct_change'].isna().all():
        df['pct_change'] = df['close_qfq'].pct_change() * 100
    df['is_limit_up'] = df['pct_change'] >= (limit_up_threshold * 100 - 0.5)
    
    # 一字板识别
    volume_ma_col = 'volume_ma20' if 'volume_ma20' in df.columns else 'v_ma5'
    avg_volume = df['volume'].mean()
    median_volume = df['volume'].median()
    df['is_one_word_limit_up'] = False
    
    for idx in df.index:
        if df.loc[idx, 'is_limit_up']:
            open_price = df.loc[idx, 'open_qfq']
            close_price = df.loc[idx, 'close_qfq']
            high_price = df.loc[idx, 'high_qfq']
            low_price = df.loc[idx, 'low_qfq']
            
            price_diff_ratio = abs(open_price - close_price) / close_price if close_price > 0 else 1.0
            is_price_match = price_diff_ratio < 0.001
            
            if not is_price_match:
                continue
            
            volume = df.loc[idx, 'volume']
            volume_ma = df.loc[idx, volume_ma_col] if pd.notna(df.loc[idx, volume_ma_col]) else avg_volume
            volume_ratio = volume / volume_ma if volume_ma > 0 else 1.0
            
            # 条件a：完全一字板（开=高=低=收）
            is_perfect_one_word = (
                abs(open_price - high_price) / close_price < 0.001 and
                abs(open_price - low_price) / close_price < 0.001 and
                abs(high_price - low_price) / close_price < 0.001
            ) if close_price > 0 else False
            
            # 条件b：成交量极小
            is_volume_tiny = (
                volume_ratio < 0.3 or 
                volume < avg_volume * 0.3 or 
                volume < median_volume * 0.5 or
                volume < 10000
            )
            
            # 条件c：准一字板（价格波动<0.5%）
            price_amplitude = abs(high_price - low_price) / close_price if close_price > 0 else 0
            is_quasi_one_word = (price_amplitude < 0.005)
            
            if is_perfect_one_word or is_volume_tiny or is_quasi_one_word:
                df.loc[idx, 'is_one_word_limit_up'] = True
    
    return df

print("="*80)
print("测试: 验证一字板识别（只统计触发日及其前5个交易日）")
print("="*80)

# 测试 002397.SZ（2025-01-02 是一字板涨停）
symbol = '002397.SZ'
trigger_date = '2025-01-02'

print(f"\n股票代码: {symbol}")
print(f"触发日期: {trigger_date}")

# 加载并处理K线数据
df_processed, trigger_dt = load_and_process_kline(symbol, trigger_date)

# 识别一字板
df_processed = identify_one_word_limit_up(df_processed, symbol)

# 找到触发日在df中的位置
trigger_idx_list = df_processed[df_processed['date'] == trigger_dt].index
if len(trigger_idx_list) > 0:
    trigger_idx = trigger_idx_list[0]
    # 获取触发日及其前5个交易日的数据（共6天）
    start_idx = max(0, trigger_idx - 5)
    end_idx = trigger_idx + 1
    df_recent = df_processed.iloc[start_idx:end_idx].copy()
    
    print(f"\n触发日在数据中的索引: {trigger_idx}")
    print(f"统计范围: 索引 {start_idx} 到 {trigger_idx}（共{end_idx - start_idx}天）")
    
    # 显示这6天的数据
    print(f"\n触发日及其前5个交易日的数据:")
    print("="*80)
    for idx, row in df_recent.iterrows():
        is_one_word = "✅ 一字板" if row['is_one_word_limit_up'] else ""
        is_limit_up = "涨停" if row['is_limit_up'] else ""
        print(f"{row['date'].strftime('%Y-%m-%d')}: "
              f"收盘价={row['close_qfq']:.2f}元（前复权）, "
              f"开盘价={row['open_qfq']:.2f}元, "
              f"涨幅={row['pct_change']:.2f}%, "
              f"成交量={row['volume']:.0f} "
              f"{is_limit_up} {is_one_word}")
    
    # 统计一字板
    one_word_days = df_recent[df_recent['is_one_word_limit_up']].copy()
    print(f"\n一字板统计:")
    print(f"  - 统计范围：触发日（{trigger_dt.strftime('%Y-%m-%d')}）及其前5个交易日（共6天）")
    print(f"  - 出现次数：{len(one_word_days)}次")
    
    if not one_word_days.empty:
        print(f"  - 出现日期：")
        for _, row in one_word_days.iterrows():
            days_before = (trigger_dt - row['date']).days
            print(f"    - {row['date'].strftime('%Y-%m-%d')}（触发日前{days_before}天）")
    
    # 检查触发日当天是否是一字板
    trigger_row = df_processed.loc[trigger_idx]
    print(f"\n触发日（{trigger_date}）数据详情:")
    print(f"  - 收盘价（前复权）: {trigger_row['close_qfq']:.2f}元")
    print(f"  - 开盘价（前复权）: {trigger_row['open_qfq']:.2f}元")
    print(f"  - 开盘价 vs 收盘价差异: {abs(trigger_row['open_qfq'] - trigger_row['close_qfq']) / trigger_row['close_qfq'] * 100:.4f}%")
    print(f"  - 涨幅: {trigger_row['pct_change']:.2f}%")
    print(f"  - 是否涨停: {'是' if trigger_row['is_limit_up'] else '否'}")
    print(f"  - 成交量: {trigger_row['volume']:.0f}")
    if 'volume_ma20' in trigger_row and pd.notna(trigger_row['volume_ma20']):
        volume_ratio = trigger_row['volume'] / trigger_row['volume_ma20']
        print(f"  - 量比（vs MA20）: {volume_ratio:.2f}")
    print(f"  - 是否一字板: {'✅ 是' if trigger_row['is_one_word_limit_up'] else '❌ 否'}")
    
    # 从JSON结果对比
    print(f"\n期望结果:")
    print(f"  - 用户确认 {trigger_date} 是一字板涨停（开盘即封板09:25:00）")
    print(f"  - 涨停价: 3.23元")
    
    if trigger_row['is_one_word_limit_up']:
        print(f"\n✅ 修复成功！触发日被正确识别为一字板")
    else:
        print(f"\n⚠️  触发日未被识别为一字板，可能原因:")
        print(f"  - 成交量判断条件可能过严（量比{volume_ratio:.2f} >= 0.3）")
        print(f"  - 需要调整一字板识别阈值")

print("\n" + "="*80)
print("测试完成")
print("="*80)
