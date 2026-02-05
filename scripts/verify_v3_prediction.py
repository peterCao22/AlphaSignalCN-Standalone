#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证V3.0预测准确性

目标：验证V3回归模型的预测涨幅与实际涨幅的相关性

时间窗口对齐：
- 预测：相对T3收盘价，预测T4+开始的5天最大涨幅
- 验证：相对T3收盘价，统计T4+开始的5天最大涨幅

使用方法：
    # 方式1：传入T3日期（推荐，与predict_v3.py保持一致）
    python scripts/verify_v3_prediction.py --date 2025-01-13 \\
        --compare results/v3_predictions/v3_prediction_T0_20250108_T3_20250113_ensemble_xxx.csv
    
    # 方式2：传入T0日期
    python scripts/verify_v3_prediction.py --t0-date 2025-01-08 \\
        --compare results/v3_predictions/v3_prediction_T0_20250108_T3_20250113_ensemble_xxx.csv

作者：AI Assistant
日期：2026-02-04
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import argparse
from datetime import datetime
from stockainews.core.logger import logger
from stockainews.services.layer3_data_supplement import Layer3DataSupplement

# 数据库配置
DB_HOST = '192.168.21.39'
DB_PORT = 5433
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'
DB_NAME = 'stocks_data'

# 结果目录
RESULTS_DIR = REPO_ROOT / 'results' / 'verification'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_limit_up_stocks_from_api(date_str):
    """从API获取涨停股池"""
    logger.info(f"\n从API获取涨停股池: {date_str}")
    
    service = Layer3DataSupplement()
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    trade_date_str = target_date.strftime('%Y%m%d')
    
    try:
        limit_up_list = service.api_adapter.get_limit_up_pool(trade_date_str)
    except Exception as e:
        msg = str(e)
        if service._is_api_limit_error(msg):
            try:
                logger.warning(f"API配额不足，切换API重试...")
                service.switch_api()
                limit_up_list = service.api_adapter.get_limit_up_pool(trade_date_str)
            except Exception as e2:
                logger.error(f"涨停池获取失败: {e2}")
                return []
        else:
            logger.error(f"涨停池获取失败: {e}")
            return []
    
    if not limit_up_list:
        logger.warning(f"未获取到涨停股票")
        return []
    
    stocks = []
    for item in limit_up_list:
        if not isinstance(item, dict):
            continue
        
        stock_code = str(
            item.get("stock_code") or 
            item.get("code") or 
            item.get("dm") or ""
        ).strip()
        
        if not (len(stock_code) == 6 and stock_code.isdigit()):
            continue
        
        if stock_code.startswith(('6', '5', '9')):
            instrument = f"{stock_code}.SH"
        else:
            instrument = f"{stock_code}.SZ"
        
        stock_name = str(
            item.get("stock_name") or 
            item.get("name") or 
            item.get("mc") or ""
        ).strip()
        
        stocks.append({
            'instrument': instrument,
            'stock_name': stock_name,
        })
    
    logger.info(f"✓ 获取到 {len(stocks)} 只涨停股票")
    return stocks


def load_trading_calendar(engine):
    """加载交易日历"""
    query = """
        SELECT DISTINCT date 
        FROM kline_all 
        WHERE date >= '2020-01-01'
        ORDER BY date
    """
    df = pd.read_sql(query, engine)
    return pd.to_datetime(df['date']).tolist()


def get_previous_trading_date(trading_days, date, n_days):
    """获取指定日期前N个交易日"""
    date = pd.Timestamp(date)
    try:
        idx = trading_days.index(date)
    except ValueError:
        for i, day in enumerate(trading_days):
            if day > date:
                idx = i - 1
                break
        else:
            return None
    
    if idx - n_days >= 0:
        return trading_days[idx - n_days]
    return None


def get_next_n_trading_dates(trading_days, start_date, n=5):
    """获取指定日期后的N个交易日"""
    start_date = pd.Timestamp(start_date)
    
    try:
        start_idx = trading_days.index(start_date)
    except ValueError:
        for i, day in enumerate(trading_days):
            if day > start_date:
                start_idx = i - 1
                break
        else:
            return []
    
    next_dates = []
    for i in range(1, n + 1):
        if start_idx + i < len(trading_days):
            next_dates.append(trading_days[start_idx + i])
        else:
            logger.warning(f"  [WARNING] 只能获取到T{i-1}，数据不足")
            break
    
    return next_dates


def get_stock_prices(engine, stock_code, start_date, end_date):
    """获取股票价格"""
    query = """
        SELECT date, close, high, low
        FROM kline_all
        WHERE instrument = %s
          AND date BETWEEN %s AND %s
        ORDER BY date
    """
    df = pd.read_sql(query, engine, params=(stock_code, start_date, end_date))
    return df


def calculate_max_gain_from_t3(prices_df, T3_close):
    """
    计算T4+期间的最大涨幅（相对T3收盘价）
    
    同时计算：
    - max_gain_3d: T4-T6（实盘持有期）
    - max_gain_5d: T4-T8（训练目标）
    
    Args:
        prices_df: T4+的价格数据
        T3_close: T3日收盘价
    
    Returns:
        dict: 包含每天涨幅和最大涨幅的字典
    """
    if len(prices_df) == 0 or T3_close <= 0:
        return {
            'T4': None, 'T5': None, 'T6': None, 'T7': None, 'T8': None,
            'max_gain_3d': None,  # 实盘持有期（T4-T6）
            'max_gain_3d_date': None,
            'max_gain_5d': None,  # 训练目标（T4-T8）
            'max_gain_5d_date': None,
            'days_count': 0
        }
    
    # 计算每天相对T3的最高涨幅
    daily_gains = {}
    max_gain_3d = -999  # T4-T6
    max_gain_3d_date = None
    max_gain_5d = -999  # T4-T8
    max_gain_5d_date = None
    
    for idx, row in prices_df.iterrows():
        day_num = idx + 4  # T4, T5, T6, T7, T8
        if day_num > 8:
            break
        
        # 当天最高价相对T3收盘价的涨幅
        gain = (row['high'] - T3_close) / T3_close * 100
        daily_gains[f'T{day_num}'] = round(gain, 2)
        
        # T4-T6（实盘持有期）
        if day_num <= 6:
            if gain > max_gain_3d:
                max_gain_3d = gain
                max_gain_3d_date = row['date']
        
        # T4-T8（训练目标）
        if gain > max_gain_5d:
            max_gain_5d = gain
            max_gain_5d_date = row['date']
    
    # 填充缺失的天数
    for i in range(4, 9):
        if f'T{i}' not in daily_gains:
            daily_gains[f'T{i}'] = None
    
    return {
        **daily_gains,
        'max_gain_3d': round(max_gain_3d, 2) if max_gain_3d > -999 else None,
        'max_gain_3d_date': max_gain_3d_date,
        'max_gain_5d': round(max_gain_5d, 2) if max_gain_5d > -999 else None,
        'max_gain_5d_date': max_gain_5d_date,
        'days_count': len(prices_df)
    }


def verify_v3_predictions(t3_date_str=None, t0_date_str=None, compare_file=None):
    """
    验证V3预测
    
    Args:
        t3_date_str: T3日期（整固结束日）
        t0_date_str: T0日期（涨停日），如果提供则自动计算T3
        compare_file: 预测结果文件路径
    """
    
    # 确定T3日期
    if t3_date_str:
        T3_date = pd.Timestamp(t3_date_str)
        mode = "T3模式"
    elif t0_date_str:
        # 需要先加载交易日历来计算T3
        mode = "T0模式（需要计算T3）"
    else:
        raise ValueError("必须提供--date（T3）或--t0-date（T0）")
    
    print("=" * 80)
    print("V3.0 预测验证脚本")
    if t3_date_str:
        print(f"T3日期（整固结束日）: {T3_date.strftime('%Y-%m-%d')}")
    else:
        print(f"T0日期（涨停日）: {t0_date_str}")
        print("  → 将自动计算T3日期（T0+3个交易日）")
    print("=" * 80)
    
    # 连接数据库
    print("\n[1/6] 连接数据库...")
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url, pool_pre_ping=True)
    print("  [OK] 连接成功")
    
    # 加载交易日历
    print("\n[2/6] 加载交易日历...")
    trading_days = load_trading_calendar(engine)
    print(f"  [OK] 加载了 {len(trading_days)} 个交易日")
    
    # 如果提供的是T0日期，计算T3
    if t0_date_str:
        T0_date = pd.Timestamp(t0_date_str)
        
        # T3 = T0 + 3个交易日
        try:
            T0_idx = trading_days.index(T0_date)
            if T0_idx + 3 < len(trading_days):
                T3_date = trading_days[T0_idx + 3]
                print(f"  [INFO] T0: {T0_date.strftime('%Y-%m-%d')} → T3: {T3_date.strftime('%Y-%m-%d')}")
            else:
                print(f"  [ERROR] 无法计算T3（T0之后交易日不足）")
                engine.dispose()
                return
        except ValueError:
            print(f"  [ERROR] {T0_date} 不是交易日")
            engine.dispose()
            return
    else:
        # 反推T0（用于获取涨停池）
        T0_date = get_previous_trading_date(trading_days, T3_date, 3)
        if T0_date is None:
            print(f"  [ERROR] 无法反推T0日期")
            engine.dispose()
            return
        print(f"  [INFO] T3: {T3_date.strftime('%Y-%m-%d')} → T0: {T0_date.strftime('%Y-%m-%d')}")
    
    # 从API获取T0的涨停池
    print(f"\n[3/6] 从API获取T0涨停池 ({T0_date.strftime('%Y-%m-%d')})...")
    stocks = get_limit_up_stocks_from_api(T0_date.strftime('%Y-%m-%d'))
    
    if len(stocks) == 0:
        print("  [ERROR] 未获取到涨停股票")
        engine.dispose()
        return
    
    print(f"  [OK] 获取到 {len(stocks)} 只股票")
    
    # 获取T4+日期（预测窗口）
    print("\n[4/6] 计算预测窗口...")
    next_dates = get_next_n_trading_dates(trading_days, T3_date, 5)
    
    if len(next_dates) == 0:
        print(f"  [ERROR] 无法获取T3之后的交易日")
        engine.dispose()
        return
    
    print(f"  [OK] 预测窗口: T4-T{3+len(next_dates)}")
    for i, date in enumerate(next_dates):
        label = " (实盘卖出)" if i == 2 else ""  # T6是实盘卖出日
        print(f"    T{i+4}: {date.strftime('%Y-%m-%d')}{label}")
    
    # 获取T3收盘价和T4+价格
    print(f"\n[5/6] 计算实际涨幅...")
    
    results = []
    start_date = T3_date.strftime('%Y-%m-%d')
    end_date = next_dates[-1].strftime('%Y-%m-%d')
    
    for idx, stock_info in enumerate(stocks):
        stock_code = stock_info['instrument']
        stock_name = stock_info['stock_name']
        
        # 获取T3到T8的价格数据
        prices_df = get_stock_prices(engine, stock_code, start_date, end_date)
        
        if len(prices_df) == 0:
            results.append({
                'instrument': stock_code,
                'stock_name': stock_name,
                'T3_close': None,
                'T4': None, 'T5': None, 'T6': None, 'T7': None, 'T8': None,
                'max_gain_5d': None,
                'max_gain_date': None,
                'days_count': 0
            })
            continue
        
        # T3收盘价
        T3_close = prices_df.iloc[0]['close'] if len(prices_df) > 0 else 0
        
        # T4+的价格数据
        prices_t4_plus = prices_df.iloc[1:] if len(prices_df) > 1 else pd.DataFrame()
        
        # 计算涨幅
        gain_info = calculate_max_gain_from_t3(prices_t4_plus, T3_close)
        
        results.append({
            'instrument': stock_code,
            'stock_name': stock_name,
            'T3_close': T3_close,
            **gain_info
        })
        
        # 显示进度
        if (idx + 1) % 10 == 0 or (idx + 1) == len(stocks):
            print(f"  进度: {idx + 1}/{len(stocks)}")
    
    engine.dispose()
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 统计结果
    print("\n[6/6] 统计结果...")
    
    valid_stocks = df_results[df_results['days_count'] > 0]
    
    if len(valid_stocks) == 0:
        print("  [ERROR] 没有有效数据")
        return
    
    print(f"  有效股票: {len(valid_stocks)}/{len(stocks)}")
    print(f"  数据完整度: T4({len(valid_stocks[valid_stocks['T4'].notna()])}), "
          f"T5({len(valid_stocks[valid_stocks['T5'].notna()])}), "
          f"T6({len(valid_stocks[valid_stocks['T6'].notna()])}), "
          f"T7({len(valid_stocks[valid_stocks['T7'].notna()])}), "
          f"T8({len(valid_stocks[valid_stocks['T8'].notna()])})")
    
    # 统计涨幅分布
    max_gains_3d = valid_stocks['max_gain_3d'].dropna()
    max_gains_5d = valid_stocks['max_gain_5d'].dropna()
    
    if len(max_gains_3d) == 0 and len(max_gains_5d) == 0:
        print("  [ERROR] 没有有效的涨幅数据")
        return
    
    # T4-T6（实盘持有期）
    if len(max_gains_3d) > 0:
        print(f"\n  【实盘持有期】T4-T6（3天）最大涨幅统计:")
        print(f"    平均值: {max_gains_3d.mean():.2f}%")
        print(f"    中位数: {max_gains_3d.median():.2f}%")
        print(f"    标准差: {max_gains_3d.std():.2f}%")
        print(f"    最小值: {max_gains_3d.min():.2f}%")
        print(f"    最大值: {max_gains_3d.max():.2f}%")
        print(f"    涨幅>10%: {len(max_gains_3d[max_gains_3d >= 10])}/{len(max_gains_3d)} ({len(max_gains_3d[max_gains_3d >= 10])/len(max_gains_3d)*100:.1f}%)")
        print(f"    涨幅>20%: {len(max_gains_3d[max_gains_3d >= 20])}/{len(max_gains_3d)} ({len(max_gains_3d[max_gains_3d >= 20])/len(max_gains_3d)*100:.1f}%)")
        print(f"    涨幅>30%: {len(max_gains_3d[max_gains_3d >= 30])}/{len(max_gains_3d)} ({len(max_gains_3d[max_gains_3d >= 30])/len(max_gains_3d)*100:.1f}%)")
    
    # T4-T8（训练目标）
    if len(max_gains_5d) > 0:
        print(f"\n  【训练目标】T4-T8（5天）最大涨幅统计:")
        print(f"    平均值: {max_gains_5d.mean():.2f}%")
        print(f"    中位数: {max_gains_5d.median():.2f}%")
        print(f"    标准差: {max_gains_5d.std():.2f}%")
        print(f"    最小值: {max_gains_5d.min():.2f}%")
        print(f"    最大值: {max_gains_5d.max():.2f}%")
        print(f"    涨幅>10%: {len(max_gains_5d[max_gains_5d >= 10])}/{len(max_gains_5d)} ({len(max_gains_5d[max_gains_5d >= 10])/len(max_gains_5d)*100:.1f}%)")
        print(f"    涨幅>20%: {len(max_gains_5d[max_gains_5d >= 20])}/{len(max_gains_5d)} ({len(max_gains_5d[max_gains_5d >= 20])/len(max_gains_5d)*100:.1f}%)")
        print(f"    涨幅>30%: {len(max_gains_5d[max_gains_5d >= 30])}/{len(max_gains_5d)} ({len(max_gains_5d[max_gains_5d >= 30])/len(max_gains_5d)*100:.1f}%)")
    
    # 对比3天vs5天
    if len(max_gains_3d) > 0 and len(max_gains_5d) > 0:
        print(f"\n  【3天 vs 5天对比】:")
        print(f"    3天平均: {max_gains_3d.mean():.2f}% vs 5天平均: {max_gains_5d.mean():.2f}%")
        print(f"    差异: {max_gains_5d.mean() - max_gains_3d.mean():.2f}% (多持有2天的额外收益)")
    
    # Top 10（按3天涨幅排序）
    if len(max_gains_3d) > 0:
        print(f"\n  Top 10 实际表现（按3天涨幅排序）:")
        top10 = valid_stocks.nlargest(10, 'max_gain_3d')[['instrument', 'stock_name', 'max_gain_3d', 'max_gain_5d', 'max_gain_3d_date']]
        for idx, row in top10.iterrows():
            gain_date = pd.Timestamp(row['max_gain_3d_date']).strftime('%m-%d') if pd.notna(row['max_gain_3d_date']) else 'N/A'
            gain_5d = row['max_gain_5d'] if pd.notna(row['max_gain_5d']) else 0
            print(f"    {idx+1:2d}. {row['instrument']:10s} {row['stock_name']:10s} "
                  f"3天: {row['max_gain_3d']:6.2f}% | 5天: {gain_5d:6.2f}% (at {gain_date})")
    
    # 如果有预测结果，进行对比
    if compare_file:
        print(f"\n" + "=" * 80)
        print("对比预测结果")
        print("=" * 80)
        
        try:
            df_predict = pd.read_csv(compare_file)
            
            if 'instrument' in df_predict.columns and 'prediction_score' in df_predict.columns:
                # 合并
                df_compare = df_results.merge(
                    df_predict[['instrument', 'prediction_score']],
                    on='instrument',
                    how='inner'
                )
                
                if len(df_compare) > 0:
                    from scipy.stats import spearmanr
                    
                    # 使用3天涨幅（实盘持有期）进行对比
                    valid_compare_3d = df_compare[df_compare['max_gain_3d'].notna() & df_compare['prediction_score'].notna()]
                    valid_compare_5d = df_compare[df_compare['max_gain_5d'].notna() & df_compare['prediction_score'].notna()]
                    
                    # 3天对比（实盘）
                    if len(valid_compare_3d) >= 3:
                        spearman_3d, p_value_3d = spearmanr(valid_compare_3d['prediction_score'], valid_compare_3d['max_gain_3d'])
                        mae_3d = np.abs(valid_compare_3d['prediction_score'] - valid_compare_3d['max_gain_3d']).mean()
                        rmse_3d = np.sqrt(((valid_compare_3d['prediction_score'] - valid_compare_3d['max_gain_3d']) ** 2).mean())
                        
                        print(f"\n【实盘持有期对比】T4-T6（3天）:")
                        print(f"  匹配股票数: {len(df_compare)}")
                        print(f"  有效对比数: {len(valid_compare_3d)}")
                        print(f"  Spearman相关系数: {spearman_3d:.4f} (p-value: {p_value_3d:.4f})")
                        print(f"  MAE (平均绝对误差): {mae_3d:.2f}%")
                        print(f"  RMSE (均方根误差): {rmse_3d:.2f}%")
                    
                    # 5天对比（训练目标）
                    if len(valid_compare_5d) >= 3:
                        spearman_5d, p_value_5d = spearmanr(valid_compare_5d['prediction_score'], valid_compare_5d['max_gain_5d'])
                        mae_5d = np.abs(valid_compare_5d['prediction_score'] - valid_compare_5d['max_gain_5d']).mean()
                        rmse_5d = np.sqrt(((valid_compare_5d['prediction_score'] - valid_compare_5d['max_gain_5d']) ** 2).mean())
                        
                        print(f"\n【训练目标对比】T4-T8（5天）:")
                        print(f"  Spearman相关系数: {spearman_5d:.4f} (p-value: {p_value_5d:.4f})")
                        print(f"  MAE (平均绝对误差): {mae_5d:.2f}%")
                        print(f"  RMSE (均方根误差): {rmse_5d:.2f}%")
                    
                    # Top 10预测对比（使用3天涨幅）
                    if len(valid_compare_3d) >= 3:
                        print(f"\nTop 10 预测对比（按预测排序）:")
                        top10_compare = valid_compare_3d.nlargest(10, 'prediction_score')[
                            ['instrument', 'stock_name', 'prediction_score', 'max_gain_3d', 'max_gain_5d']
                        ].reset_index(drop=True)
                        
                        for idx, row in top10_compare.iterrows():
                            diff_3d = row['max_gain_3d'] - row['prediction_score']
                            gain_5d = row['max_gain_5d'] if pd.notna(row['max_gain_5d']) else 0
                            symbol = "OK" if abs(diff_3d) < 10 else "X"
                            print(f"  {idx+1:2d}. {row['instrument']:10s} {row['stock_name']:10s} "
                                  f"预测: {row['prediction_score']:6.2f}% -> 实际3天: {row['max_gain_3d']:6.2f}% "
                                  f"(差: {diff_3d:+.2f}%) [{symbol}] | 5天: {gain_5d:6.2f}%")
                    else:
                        print(f"[WARNING] 有效对比数据不足")
                else:
                    print(f"[WARNING] 没有匹配的股票")
            else:
                print(f"[WARNING] 预测文件格式不正确")
                
        except Exception as e:
            print(f"[ERROR] 对比失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f'v3_verification_T0_{T0_date.strftime("%Y%m%d")}_T3_{T3_date.strftime("%Y%m%d")}_{timestamp}.csv'
    
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 结果已保存: {output_file}")
    
    print("\n" + "=" * 80)
    print("[COMPLETED] 验证完成")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='验证V3.0预测准确性')
    parser.add_argument('--date', help='T3日期（整固结束日），格式: YYYY-MM-DD')
    parser.add_argument('--t0-date', help='T0日期（涨停日），格式: YYYY-MM-DD')
    parser.add_argument('--compare', help='预测结果CSV文件路径')
    
    args = parser.parse_args()
    
    if not args.date and not args.t0_date:
        parser.error("必须提供--date（T3）或--t0-date（T0）")
    
    verify_v3_predictions(
        t3_date_str=args.date,
        t0_date_str=args.t0_date,
        compare_file=args.compare
    )


if __name__ == '__main__':
    main()
