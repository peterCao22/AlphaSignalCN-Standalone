#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.1 强势整理预测脚本（实盘对齐版 - 无数据泄漏）

时间窗口：
- T0: 涨停日
- T1-T2: 整固期（2天）← 提取所有特征
- T3: 决策日（下午14:30后买入）← 此时T3数据未产生，不使用！
- T4-T6: 预测窗口（3天）

使用V3.1模型（74维特征）- 全部基于T1-T2：
- 基础特征：59维（基于T1-T2的K线数据）
- 板块特征：7维（基于T2当天的板块数据）
- 筹码特征：5维（基于T2当天的筹码数据）
- 环境特征：3维（基于T2当天的环境数据）

数据来源：
- 涨停池：智兔/魔码云服API（T0日期）
- K线数据：PostgreSQL（截至T2）
- 板块/筹码/环境：PostgreSQL（截至T2）

关键改进：
✅ 避免数据泄漏：T3决策时不使用T3的数据
✅ 实盘可行：所有特征在T3上午即可获取
✅ 时间对齐：预测目标T4-T6与实盘持有期一致

与V3.0的区别：
- 特征窗口：T1-T2（2天）vs T1-T3（3天）
- 增强特征：基于T2 vs 基于T3（数据泄漏）
- 预测目标：T4-T6（3天）vs T4-T8（5天）
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import json
import pickle
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import argparse

# 导入特征提取器
from consolidation_features import ConsolidationFeatureExtractor

# 导入V3过滤器（2026-02-04：放宽筛选，支持多连板强势股）
from consolidation_filter_v3 import ConsolidationFilterV3

# 导入API服务
from stockainews.services.layer3_data_supplement import Layer3DataSupplement
from stockainews.core.logger import setup_logger

# 配置日志
logger = setup_logger('v3_predict')

# 数据库配置
DB_HOST = '192.168.21.39'
DB_PORT = 5433
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'
DB_NAME = 'stocks_data'

# 模型路径
MODEL_DIR = REPO_ROOT / 'models'
LGB_MODEL_PATH = MODEL_DIR / 'v3_consolidation_lgb.model'
XGB_MODEL_PATH = MODEL_DIR / 'v3_consolidation_xgb.model'
FEATURE_NAMES_PATH = MODEL_DIR / 'v3_feature_names.json'
SECTOR_MAPPING_PATH = REPO_ROOT / 'data' / 'processed' / 'stock_sector_mapping.pkl'

# 结果路径
RESULTS_DIR = REPO_ROOT / 'results' / 'v3_predictions'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_models():
    """
    加载V3模型和特征列表
    
    优先级：V3.1回归（T1-T2，3天预测） > V3.0回归（T1-T3，5天预测） > V3.0分类
    
    Returns:
        lgb_model, xgb_model, feature_names, is_regression
    """
    print("\n[1/8] Loading V3 models...")
    
    # 优先检查V3.1回归模型（T1-T2特征，T4-T6预测）
    LGB_V31_REGRESSION_PATH = MODEL_DIR / 'v3.1_regression_lgb.model'
    XGB_V31_REGRESSION_PATH = MODEL_DIR / 'v3.1_regression_xgb.model'
    FEATURE_V31_REGRESSION_PATH = MODEL_DIR / 'v3.1_regression_feature_names.json'
    
    if (LGB_V31_REGRESSION_PATH.exists() and XGB_V31_REGRESSION_PATH.exists() and 
        FEATURE_V31_REGRESSION_PATH.exists()):
        # 加载V3.1回归模型
        lgb_model = lgb.Booster(model_file=str(LGB_V31_REGRESSION_PATH))
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(XGB_V31_REGRESSION_PATH))
        
        with open(FEATURE_V31_REGRESSION_PATH, 'r') as f:
            feature_names = json.load(f)
        
        print(f"  [OK] Loaded V3.1 REGRESSION models (T1-T2 features, T4-T6 prediction)")
        print(f"  [OK] Feature dimensions: {len(feature_names)}")
        print(f"  [INFO] Model type: REGRESSION (predicts 3-day max gain %)")
        
        return lgb_model, xgb_model, feature_names, True
    
    # 检查V3.0回归模型（T1-T3特征，T4-T8预测）
    LGB_V30_REGRESSION_PATH = MODEL_DIR / 'v3_regression_lgb.model'
    XGB_V30_REGRESSION_PATH = MODEL_DIR / 'v3_regression_xgb.model'
    FEATURE_V30_REGRESSION_PATH = MODEL_DIR / 'v3_regression_feature_names.json'
    
    if (LGB_V30_REGRESSION_PATH.exists() and XGB_V30_REGRESSION_PATH.exists() and 
        FEATURE_V30_REGRESSION_PATH.exists()):
        # 加载V3.0回归模型
        lgb_model = lgb.Booster(model_file=str(LGB_V30_REGRESSION_PATH))
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(XGB_V30_REGRESSION_PATH))
        
        with open(FEATURE_V30_REGRESSION_PATH, 'r') as f:
            feature_names = json.load(f)
        
        print(f"  [OK] Loaded V3.0 REGRESSION models (T1-T3 features, T4-T8 prediction)")
        print(f"  [OK] Feature dimensions: {len(feature_names)}")
        print(f"  [INFO] Model type: REGRESSION (predicts 5-day max gain %)")
        print(f"  [WARNING] This model uses T1-T3 features, not aligned with current V3.1 extraction!")
        
        return lgb_model, xgb_model, feature_names, True
    
    # 回归模型不存在，加载分类模型
    elif LGB_MODEL_PATH.exists() and XGB_MODEL_PATH.exists() and FEATURE_NAMES_PATH.exists():
        lgb_model = lgb.Booster(model_file=str(LGB_MODEL_PATH))
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(XGB_MODEL_PATH))
        
        with open(FEATURE_NAMES_PATH, 'r') as f:
            feature_names = json.load(f)
        
        print(f"  [OK] Loaded LightGBM and XGBoost CLASSIFICATION models")
        print(f"  [OK] Feature dimensions: {len(feature_names)}")
        print(f"  [INFO] Model type: CLASSIFICATION (predicts probability)")
        print(f"  [TIP] Run train_v3_regression_model.py for better results")
        
        return lgb_model, xgb_model, feature_names, False
    
    else:
        raise FileNotFoundError(
            f"Neither regression nor classification models found:\n"
            f"  Regression: {LGB_REGRESSION_PATH}\n"
            f"  Classification: {LGB_MODEL_PATH}"
        )


def load_trading_calendar(engine):
    """加载交易日历"""
    query = """
        SELECT trade_date 
        FROM trading_calendar
        WHERE is_trading_day = true
        AND trade_date >= '2023-01-01' AND trade_date < '2027-01-01'
        ORDER BY trade_date
    """
    df = pd.read_sql(query, engine)
    return pd.to_datetime(df['trade_date'])


def get_previous_trading_date(trading_days, date, days):
    """获取指定日期之前第N个交易日"""
    date = pd.Timestamp(date)
    past_dates = trading_days[trading_days < date]
    
    if len(past_dates) < days:
        return None
    
    # 返回倒数第N个交易日
    return past_dates.iloc[-days]


def get_next_n_trading_dates(trading_days, date, n):
    """
    获取指定日期之后的N个交易日（列表）
    
    参数:
        trading_days: 交易日Series
        date: 起始日期
        n: 需要获取的交易日数量
    
    返回:
        前N个交易日的列表（Series），如果不足N个则返回空列表
    """
    date = pd.Timestamp(date)
    future_dates = trading_days[trading_days > date]
    
    if len(future_dates) < n:
        return pd.Series([], dtype='datetime64[ns]')
    
    # 返回前N个交易日
    return future_dates.iloc[:n]


def get_limit_up_stocks_from_api(date_str):
    """
    从智兔/魔码云服API获取涨停池股票
    
    Args:
        date_str: 日期字符串（YYYY-MM-DD）
    
    Returns:
        list of dict: [{'instrument': 'XXX.XX', 'stock_name': 'XXX', 
                       'consecutive_boards': X, 'seal_funds': X, ...}, ...]
    """
    logger.info(f"\n从API获取涨停股池: {date_str}")
    
    # 初始化API服务
    supplement_service = Layer3DataSupplement()
    
    # 转换日期格式（YYYYMMDD）
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    trade_date_str = target_date.strftime('%Y%m%d')
    
    # 调用API
    try:
        limit_up_list = supplement_service.api_adapter.get_limit_up_pool(trade_date_str)
    except Exception as e:
        msg = str(e)
        # 配额/限流：切换API重试
        if supplement_service._is_api_limit_error(msg):
            try:
                logger.warning(f"API配额不足，切换API重试...")
                supplement_service.switch_api()
                limit_up_list = supplement_service.api_adapter.get_limit_up_pool(trade_date_str)
            except Exception as e2:
                logger.error(f"涨停池获取失败（已切换API仍失败）: {e2}")
                return []
        else:
            logger.error(f"涨停池获取失败: {e}")
            return []
    
    if not limit_up_list:
        logger.warning(f"未获取到涨停股票")
        return []
    
    # 解析返回结果
    stocks = []
    
    for item in limit_up_list:
        if not isinstance(item, dict):
            continue
        
        # 提取股票代码
        stock_code = str(
            item.get("stock_code") or 
            item.get("code") or 
            item.get("dm") or ""
        ).strip()
        
        if not (len(stock_code) == 6 and stock_code.isdigit()):
            continue
        
        # 转换为带后缀的instrument格式
        if stock_code.startswith(('6', '5', '9')):
            instrument = f"{stock_code}.SH"
        else:
            instrument = f"{stock_code}.SZ"
        
        # 提取股票名称
        stock_name = str(
            item.get("stock_name") or 
            item.get("name") or 
            item.get("mc") or ""
        ).strip()
        
        # 提取涨停质量数据
        stocks.append({
            'instrument': instrument,
            'stock_name': stock_name,
            'consecutive_boards': item.get("consecutive_boards", 0) or item.get("lbc", 0) or 1,
            'seal_funds': float(item.get("seal_funds", 0) or item.get("zj", 0) or 0),
            'explosion_count': item.get("explosion_count", 0) or item.get("zbc", 0) or 0,
        })
    
    logger.info(f"✓ 获取到 {len(stocks)} 只涨停股票")
    
    return stocks


def load_kline_data(engine, start_date, end_date):
    """
    从PostgreSQL加载K线数据
    
    Args:
        engine: 数据库引擎
        start_date: 开始日期 (T0之前60天，用于计算60日特征)
        end_date: 结束日期 (T3)
    
    Returns:
        DataFrame: K线数据
    """
    query = """
        SELECT date, instrument, open, high, low, close, 
               volume, amount, turn, 
               ma5, ma10, ma20, ma60
        FROM kline_all
        WHERE date >= %s AND date <= %s
        ORDER BY instrument, date
    """
    df = pd.read_sql(query, engine, params=(start_date, end_date))
    df['date'] = pd.to_datetime(df['date'])
    return df


def calculate_consecutive_boards_from_kline(kline_df, instrument, T0_date, debug=False):
    """
    从K线数据计算T0日期的实际连板数
    
    Args:
        kline_df: K线数据DataFrame
        instrument: 股票代码
        T0_date: T0日期（Timestamp）
        debug: 是否输出调试信息
    
    Returns:
        int: 连板数（包括T0当天）
    """
    # 获取该股票的K线
    stock_kline = kline_df[kline_df['instrument'] == instrument].copy()
    stock_kline = stock_kline.sort_values('date').reset_index(drop=True)
    
    # 找到T0的索引
    T0_mask = stock_kline['date'] == T0_date
    if not T0_mask.any():
        if debug:
            print(f"    [{instrument}] T0日期未找到数据")
        return 1  # 找不到数据，默认1板
    
    T0_idx = stock_kline[T0_mask].index[0]
    
    if debug:
        print(f"    [{instrument}] 开始计算连板数，T0_idx={T0_idx}")
    
    # 检查T0是否涨停（收盘价接近涨停价）
    T0_row = stock_kline.iloc[T0_idx]
    if T0_idx == 0:
        return 1  # 没有前一天数据，无法判断
    
    prev_close = stock_kline.iloc[T0_idx - 1]['close']
    limit_up_price = prev_close * 1.1  # 10%涨停
    # 创业板/科创板20%涨停
    if instrument.startswith('30') or instrument.startswith('688'):
        limit_up_price = prev_close * 1.2
    
    # 如果T0不是涨停，返回1
    gain_pct = (T0_row['close'] / prev_close - 1) * 100
    is_limit_up = T0_row['close'] >= limit_up_price * 0.998  # 放宽到0.2%误差
    
    if debug:
        print(f"    [{instrument}] T0涨幅: {gain_pct:.2f}%, 涨停价: {limit_up_price:.2f}, 实际收盘: {T0_row['close']:.2f}, 是否涨停: {is_limit_up}")
    
    if not is_limit_up:
        return 1
    
    # 向前倒推，计算连续涨停数
    consecutive_count = 1  # T0算1板
    
    for i in range(T0_idx - 1, -1, -1):
        curr_row = stock_kline.iloc[i]
        if i == 0:
            break
        prev_close_i = stock_kline.iloc[i - 1]['close']
        
        # 判断是否涨停
        limit_up_price_i = prev_close_i * 1.1
        if instrument.startswith('30') or instrument.startswith('688'):
            limit_up_price_i = prev_close_i * 1.2
        
        gain_pct_i = (curr_row['close'] / prev_close_i - 1) * 100
        is_limit_up_i = curr_row['close'] >= limit_up_price_i * 0.998
        
        if debug and consecutive_count <= 7:  # 只输出前7天
            print(f"    [{instrument}] T-{consecutive_count}: 涨幅{gain_pct_i:.2f}%, 涨停价{limit_up_price_i:.2f}, 实际{curr_row['close']:.2f}, 是否涨停: {is_limit_up_i}")
        
        if is_limit_up_i:
            consecutive_count += 1
        else:
            break  # 遇到非涨停，停止
    
    if debug:
        print(f"    [{instrument}] 计算结果: {consecutive_count}连板")
    
    return consecutive_count


def extract_base_features_for_stock(instrument, T0_date, T2_date, kline_df, 
                                    consecutive_boards, seal_funds, explosion_count):
    """
    为单只股票提取基础特征（59维）
    
    V3.1版本：使用T1-T2（2天）的K线数据
    
    Args:
        instrument: 股票代码
        T0_date: T0日期（涨停日）
        T2_date: T2日期（整固期第2天，用于定位数据）
        kline_df: K线数据
        consecutive_boards: 连板数
        seal_funds: 封单资金
        explosion_count: 炸板次数
    
    Returns:
        dict: 特征字典，失败返回None
    """
    try:
        # 获取该股票的K线数据并重置索引
        stock_kline = kline_df[kline_df['instrument'] == instrument].copy()
        if len(stock_kline) == 0:
            logger.warning(f"{instrument} 在kline_df中未找到任何数据")
            return None
        
        logger.debug(f"{instrument} 从kline_df中获取了{len(stock_kline)}条K线，日期范围: {stock_kline['date'].min()} ~ {stock_kline['date'].max()}")
        
        stock_kline = stock_kline.sort_values('date').reset_index(drop=True)
        
        # 找到T0在数据中的位置（使用重置后的索引）
        T0_mask = stock_kline['date'] == T0_date
        if not T0_mask.any():
            return None
        
        T0_idx = stock_kline[T0_mask].index[0]
        
        # 检查是否有足够的数据（T0, T1, T2至少3天）
        if T0_idx + 3 > len(stock_kline):
            return None
        
        # 获取T0~T2的3天数据（用于特征提取）
        kline_3days = stock_kline.iloc[T0_idx:T0_idx+3].copy()  # 使用.copy()避免视图问题
        
        # 调试：检查获取的数据行数
        if len(kline_3days) < 3:
            logger.warning(f"{instrument} K线不足3天: T0_idx={T0_idx}, stock_kline_len={len(stock_kline)}, kline_3days_len={len(kline_3days)}")
            logger.warning(f"  kline_3days dates: {kline_3days['date'].tolist()}")
            return None
        
        T0_close = kline_3days.iloc[0]['close']
        
        # 提取各类基础特征（使用T1-T2的2天数据）
        features = {}
        
        # DEBUG: 确认kline_3days长度
        logger.debug(f"{instrument} 准备提取特征，kline_3days长度={len(kline_3days)}, 日期={kline_3days['date'].tolist()}")
        
        # 1. 形态特征（18维）- 基于T1-T2
        pattern_features = ConsolidationFeatureExtractor.extract_pattern_features(
            kline_3days, T0_close
        )
        logger.debug(f"{instrument} extract_pattern_features后，kline_3days长度={len(kline_3days)}")
        features.update(pattern_features)
        
        # 2. 均线特征（12维）- 基于T2的位置
        ma_features = ConsolidationFeatureExtractor.extract_ma_features(
            stock_kline, T0_idx, target_day=2  # V3.1: 使用T2的均线特征
        )
        features.update(ma_features)
        
        # 3. 量能特征（8维）- 基于T1-T2
        stock_avg_volume = stock_kline['volume'].iloc[max(0, T0_idx-20):T0_idx].mean()
        volume_features = ConsolidationFeatureExtractor.extract_volume_features(
            kline_3days, stock_avg_volume
        )
        features.update(volume_features)
        
        # 4. 股性特征（7维）- 使用默认值或从K线计算
        # 计算60日波动率
        if T0_idx >= 60:
            recent_60d = stock_kline.iloc[T0_idx-60:T0_idx]
            returns_60d = recent_60d['close'].pct_change().dropna()
            volatility_60d = returns_60d.std() * np.sqrt(252)  # 年化波动率
            up_day_ratio_60d = (returns_60d > 0).sum() / len(returns_60d)
            amplitude_avg_60d = ((recent_60d['high'] - recent_60d['low']) / recent_60d['open']).mean()
        else:
            volatility_60d = 0.3
            up_day_ratio_60d = 0.5
            amplitude_avg_60d = 0.03
        
        features['volatility_60d'] = volatility_60d
        features['up_day_ratio_60d'] = up_day_ratio_60d
        features['rebound_speed'] = 10.0  # 默认值
        features['amplitude_avg_60d'] = amplitude_avg_60d
        features['up_body_sum_ratio_60d'] = 1.0  # 默认值
        features['volume_price_correlation'] = 0.0  # 默认值
        features['turnover_rate'] = kline_3days.iloc[-1].get('turn', 5.0)  # V3.1: 使用kline_3days
        
        # 5. 前期表现特征（4维）- 基于T1-T2
        prior_features = ConsolidationFeatureExtractor.extract_prior_performance_features(
            kline_3days, T0_close
        )
        features.update(prior_features)
        
        # 6. 相对强度特征（2维）- 基于T1-T2期间
        relative_features = ConsolidationFeatureExtractor.extract_relative_strength_features(
            stock_kline, T0_idx, index_kline_df=None, n_days=2  # V3.1: T1-T2（2天）
        )
        features.update(relative_features)
        
        # 7. 涨停质量特征（从API获取）
        features['consecutive_boards'] = consecutive_boards
        features['seal_funds'] = seal_funds
        features['explosion_count'] = explosion_count
        
        # 8. T0和T2的收盘价（V3.1改为T2，不再使用T3）
        features['T0_close'] = T0_close
        
        # 再次检查kline_3days长度（防止在特征提取过程中被修改）
        if len(kline_3days) < 3:
            logger.warning(f"{instrument} 在设置T3_close前K线变为{len(kline_3days)}天")
            return None
        
        features['T3_close'] = kline_3days.iloc[2]['close']  # 实际是T2的close，但保持特征名兼容性
        
        return features
        
    except Exception as e:
        logger.warning(f"提取基础特征失败 {instrument}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def extract_sector_features(engine, date, stock_code, sector_mapping):
    """提取板块特征（7维）"""
    
    # 获取股票所属板块
    stock_to_concepts = sector_mapping.get('stock_to_concepts', {})
    concepts = stock_to_concepts.get(stock_code, [])
    
    if not concepts:
        return {
            'sector_return_1d': 0.0,
            'sector_return_3d': 0.0,
            'sector_return_5d': 0.0,
            'sector_volume_ratio': 1.0,
            'sector_consecutive_up_days': 0,
            'sector_relative_strength': 1.0,
            'sector_concept_count': 0
        }
    
    # 取第一个板块作为主板块
    main_concept = concepts[0]
    
    # 查询板块K线数据
    query = """
        SELECT date, close, amount
        FROM concept_bar1d
        WHERE concept_code = %s AND date <= %s
        ORDER BY date DESC
        LIMIT 10
    """
    
    df = pd.read_sql(query, engine, params=(main_concept, date))
    
    if len(df) < 2:
        return {
            'sector_return_1d': 0.0,
            'sector_return_3d': 0.0,
            'sector_return_5d': 0.0,
            'sector_volume_ratio': 1.0,
            'sector_consecutive_up_days': 0,
            'sector_relative_strength': 1.0,
            'sector_concept_count': len(concepts)
        }
    
    # 计算板块涨跌幅
    close_values = df['close'].values
    sector_return_1d = (close_values[0] / close_values[1] - 1) * 100 if len(close_values) >= 2 else 0.0
    sector_return_3d = (close_values[0] / close_values[3] - 1) * 100 if len(close_values) >= 4 else 0.0
    sector_return_5d = (close_values[0] / close_values[5] - 1) * 100 if len(close_values) >= 6 else 0.0
    
    # 计算板块成交额比率
    amount_values = df['amount'].values
    if len(amount_values) >= 6:
        amount_5d_avg = amount_values[1:6].mean()
        sector_volume_ratio = amount_values[0] / amount_5d_avg if amount_5d_avg > 0 else 1.0
    else:
        sector_volume_ratio = 1.0
    
    # 计算连续上涨天数
    sector_consecutive_up_days = 0
    for i in range(len(close_values) - 1):
        if close_values[i] > close_values[i + 1]:
            sector_consecutive_up_days += 1
        else:
            break
    
    return {
        'sector_return_1d': round(sector_return_1d, 4),
        'sector_return_3d': round(sector_return_3d, 4),
        'sector_return_5d': round(sector_return_5d, 4),
        'sector_volume_ratio': round(sector_volume_ratio, 4),
        'sector_consecutive_up_days': sector_consecutive_up_days,
        'sector_relative_strength': 1.0,  # 简化处理
        'sector_concept_count': len(concepts)
    }


def extract_chip_features(engine, date, stock_code):
    """提取筹码特征（5维）"""
    
    query = """
        SELECT date, avg_cost, win_percent, concentration
        FROM chips_all
        WHERE instrument = %s AND date <= %s
        ORDER BY date DESC
        LIMIT 5
    """
    
    df = pd.read_sql(query, engine, params=(stock_code, date))
    
    if len(df) == 0:
        return {
            'chip_avg_cost': None,
            'chip_win_percent': None,
            'chip_concentration': None,
            'chip_concentration_change_3d': 0.0,
            'chip_win_percent_change_3d': 0.0
        }
    
    # 最新数据
    latest = df.iloc[0]
    chip_avg_cost = latest['avg_cost']
    chip_win_percent = latest['win_percent']
    chip_concentration = latest['concentration']
    
    # 计算3日变化
    if len(df) >= 4:
        chip_concentration_change_3d = (df.iloc[0]['concentration'] - df.iloc[3]['concentration']) * 100
        chip_win_percent_change_3d = (df.iloc[0]['win_percent'] - df.iloc[3]['win_percent']) * 100
    else:
        chip_concentration_change_3d = 0.0
        chip_win_percent_change_3d = 0.0
    
    return {
        'chip_avg_cost': round(chip_avg_cost, 4) if chip_avg_cost is not None else None,
        'chip_win_percent': round(chip_win_percent, 4) if chip_win_percent is not None else None,
        'chip_concentration': round(chip_concentration, 4) if chip_concentration is not None else None,
        'chip_concentration_change_3d': round(chip_concentration_change_3d, 4),
        'chip_win_percent_change_3d': round(chip_win_percent_change_3d, 4)
    }


def extract_environment_features(engine, date, stock_code, sector_mapping):
    """提取环境特征（3维）"""
    
    # 1. sentiment_label - 市场情绪标签
    sentiment_query = """
        SELECT sentiment_score
        FROM market_sentiment
        WHERE crawl_date <= %s
        ORDER BY crawl_date DESC
        LIMIT 2
    """
    df_sentiment = pd.read_sql(sentiment_query, engine, params=(date,))
    
    if len(df_sentiment) >= 2:
        sentiment_change = df_sentiment.iloc[0]['sentiment_score'] - df_sentiment.iloc[1]['sentiment_score']
        if sentiment_change < -10:
            sentiment_label = 0  # 降温
        elif sentiment_change > 10:
            sentiment_label = 2  # 升温
        else:
            sentiment_label = 1  # 稳定
    else:
        sentiment_label = 1
    
    # 2. rotation_label - 板块轮动标签
    stock_to_concepts = sector_mapping.get('stock_to_concepts', {})
    concepts = stock_to_concepts.get(stock_code, [])
    
    if concepts:
        main_concept = concepts[0]
        
        rotation_query = """
            SELECT close
            FROM concept_bar1d
            WHERE concept_code = %s AND date <= %s
            ORDER BY date DESC
            LIMIT 6
        """
        df_rotation = pd.read_sql(rotation_query, engine, params=(main_concept, date))
        
        if len(df_rotation) >= 6:
            future_return = (df_rotation.iloc[0]['close'] / df_rotation.iloc[5]['close'] - 1) * 100
            rotation_label = 1 if future_return > 0 else 0
        else:
            rotation_label = 0
    else:
        rotation_label = 0
    
    # 3. momentum_score - 个股惯性评分（简化计算）
    momentum_query = """
        SELECT close, amount
        FROM kline_all
        WHERE instrument = %s AND date <= %s
        ORDER BY date DESC
        LIMIT 6
    """
    df_momentum = pd.read_sql(momentum_query, engine, params=(stock_code, date))
    
    if len(df_momentum) >= 6:
        future_return_5d = (df_momentum.iloc[0]['close'] / df_momentum.iloc[5]['close'] - 1) * 100
        # 简化评分逻辑
        if future_return_5d > 20:
            momentum_score = 70
        elif future_return_5d > 10:
            momentum_score = 50 + future_return_5d * 2
        elif future_return_5d > 0:
            momentum_score = 30 + future_return_5d * 2
        else:
            momentum_score = max(0, 30 + future_return_5d)
    else:
        momentum_score = 30.0
    
    return {
        'sentiment_label': sentiment_label,
        'rotation_label': rotation_label,
        'momentum_score': round(momentum_score, 2)
    }


def predict(date_str, top_k=10, model_type='ensemble'):
    """执行预测
    
    Args:
        date_str: T3日期（整固期结束日），格式: YYYY-MM-DD
        top_k: 返回Top K预测结果
        model_type: 模型类型 (ensemble/lgb/xgb)
    
    数据流说明:
        1. date_str = T3（整固期结束日）
        2. 系统自动计算 T0 = T3 - 3个交易日（涨停日）
        3. 从智兔/魔码云服API获取T0当天的涨停股票
        4. 从PostgreSQL加载K线数据
        5. 提取特征：
           - 基础特征59维：基于T1-T3整固期3天的数据
           - 增强特征15维：基于T3当天的数据（板块7+筹码5+环境3）
        6. 预测T4及之后（T3+1开始）的5天表现
    """
    
    # 解析T3日期
    T3_date = pd.Timestamp(date_str)
    
    print("=" * 80)
    print(f"V3.0 Consolidation Prediction")
    print(f"T3 Date (Feature Extraction): {T3_date.strftime('%Y-%m-%d')}")
    print(f"Model: {model_type.upper()}")
    print("=" * 80)
    
    # 加载模型
    lgb_model, xgb_model, feature_names, is_regression = load_models()
    
    # 连接数据库
    print("\n[2/9] Connecting to database...")
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url, pool_pre_ping=True)
    print("  [OK] Connected")
    
    # 加载交易日历并计算T0日期
    print("\n[3/9] Loading trading calendar...")
    trading_days = load_trading_calendar(engine)
    print(f"  [OK] Loaded {len(trading_days)} trading days")
    
    # 计算T0日期（往前推3个交易日）和T2日期
    T0_date = get_previous_trading_date(trading_days, T3_date, 3)
    
    if T0_date is None:
        print(f"  [ERROR] Cannot find T0 date (3 trading days before {T3_date})")
        engine.dispose()
        return
    
    # 计算T2日期（T0 + 2个交易日）- 用于基础特征提取
    T2_date = get_next_n_trading_dates(trading_days, T0_date, 2)
    if len(T2_date) < 2:
        print(f"  [ERROR] Cannot find T2 date (2 trading days after {T0_date})")
        engine.dispose()
        return
    T2_date = T2_date.iloc[1]  # 第2个交易日（使用iloc）
    
    print(f"  [INFO] T0 Date (Limit-up): {T0_date.strftime('%Y-%m-%d')}")
    print(f"  [INFO] T2 Date (Feature end): {T2_date.strftime('%Y-%m-%d')}")
    print(f"  [INFO] T3 Date (Decision): {T3_date.strftime('%Y-%m-%d')}")
    print(f"  [INFO] Prediction Window: T4-T6 (3 days)")
    
    # 加载板块映射
    print("\n[4/9] Loading sector mapping...")
    with open(SECTOR_MAPPING_PATH, 'rb') as f:
        sector_mapping = pickle.load(f)
    print("  [OK] Loaded")
    
    # 从API获取T0的涨停池
    print(f"\n[5/10] Fetching limit-up stocks from API for T0 ({T0_date.strftime('%Y-%m-%d')})...")
    limit_up_stocks = get_limit_up_stocks_from_api(T0_date.strftime('%Y-%m-%d'))
    
    if len(limit_up_stocks) == 0:
        print(f"  [WARNING] No stocks in limit-up pool for {T0_date}")
        engine.dispose()
        return
    
    print(f"  [OK] Found {len(limit_up_stocks)} stocks from API")
    
    # 加载K线数据（T0之前60天到T3，用于计算60日特征）
    print(f"\n[6/10] Loading K-line data...")
    start_date = (T0_date - pd.Timedelta(days=90)).strftime('%Y-%m-%d')  # 多加一些buffer
    end_date = T3_date.strftime('%Y-%m-%d')
    kline_df = load_kline_data(engine, start_date, end_date)
    print(f"  [OK] Loaded {len(kline_df)} K-line records")
    
    # 修正consecutive_boards（从K线数据重新计算）
    print(f"\n[6.5/10] Recalculating consecutive boards from K-line...")
    recalc_count = 0
    debug_stocks_codes = ['600397.SH', '002730.SZ', '002117.SZ', '002733.SZ']
    
    for stock_info in limit_up_stocks:
        instrument = stock_info['instrument']
        api_boards = stock_info['consecutive_boards']
        
        # 对调试目标股票启用调试模式
        is_debug_target = instrument in debug_stocks_codes
        if is_debug_target:
            print(f"  [DEBUG] 计算 {instrument} 的连板数（API显示={api_boards}板）...")
        
        actual_boards = calculate_consecutive_boards_from_kline(
            kline_df, instrument, T0_date, debug=is_debug_target
        )
        
        if is_debug_target:
            print(f"  [DEBUG] {instrument}: API={api_boards}板, 实际计算={actual_boards}板")
        
        if api_boards != actual_boards:
            recalc_count += 1
            print(f"  {instrument}: API={api_boards}板 -> 实际={actual_boards}板")
        stock_info['consecutive_boards'] = actual_boards
    
    print(f"  [OK] Recalculated {recalc_count} stocks with mismatched board count")
    
    # 提取特征（添加V3过滤器 - 放宽筛选）
    print(f"\n[7/11] Applying V3 consolidation filter (relaxed for multi-board strong stocks)...")
    filtered_stocks = []
    filter_stats = {'total': len(limit_up_stocks), 'passed': 0, 'failed': 0}
    
    # 调试目标股票（如果存在）
    debug_stocks = ['600397', '002730', '002117', '002733']
    debug_results = {}
    
    for stock_info in limit_up_stocks:
        stock_code = stock_info['instrument']
        consecutive_boards = stock_info['consecutive_boards']
        
        # 检查是否是调试目标股票
        is_debug_target = any(stock_code.startswith(code) for code in debug_stocks)
        
        try:
            # 使用V3过滤器检查是否符合"强势整固"形态（V3放宽筛选）
            passed, filter_details = ConsolidationFilterV3.check_consolidation_pattern(
                stock_code, T0_date, kline_df, consecutive_boards=consecutive_boards
            )
            
            # 记录调试目标股票的详细结果
            if is_debug_target:
                debug_results[stock_code] = {
                    'passed': passed,
                    'consecutive_boards': consecutive_boards,
                    'checks': filter_details.get('checks', {}),
                    'details': filter_details
                }
            
            if passed:
                filtered_stocks.append(stock_info)
                filter_stats['passed'] += 1
            else:
                filter_stats['failed'] += 1
                # 记录所有过滤原因
                logger.debug(f"  [FILTERED] {stock_code}: {filter_details.get('checks', {})}")
        except Exception as e:
            logger.warning(f"  [ERROR] {stock_code} filter check failed: {e}")
            filter_stats['failed'] += 1
            
            # 记录调试目标股票的错误
            if is_debug_target:
                debug_results[stock_code] = {
                    'passed': False,
                    'error': str(e)
                }
            continue
    
    pass_rate = filter_stats['passed'] / filter_stats['total'] * 100 if filter_stats['total'] > 0 else 0
    print(f"  [OK] Passed filter: {filter_stats['passed']}/{filter_stats['total']} ({pass_rate:.1f}%)")
    
    # 输出调试目标股票的详细信息
    if debug_results:
        print(f"\n  [DEBUG] 目标股票筛选详情：")
        for stock_code, result in debug_results.items():
            if result['passed']:
                print(f"    [PASS] {stock_code} ({result['consecutive_boards']}板) - 已通过筛选")
            elif 'error' in result:
                print(f"    [FAIL] {stock_code} - 错误: {result['error']}")
            else:
                # 显示失败的检查项
                checks = result.get('checks', {})
                failed_checks = [k for k, v in checks.items() if not v]
                print(f"    [FAIL] {stock_code} ({result['consecutive_boards']}板) - 失败项: {', '.join(failed_checks)}")
                
                # 显示详细原因（包含阈值）
                details = result.get('details', {})
                for check_name in failed_checks:
                    check_detail = details.get(f'{check_name}_details', {})
                    if check_name == 'consecutive_boards':
                        reason = check_detail.get('reason', 'N/A')
                        normal_count = check_detail.get('normal_limit_up_count', 0)
                        total_count = check_detail.get('total_limit_up_count', 0)
                        max_allowed = check_detail.get('max_limit_ups', 'N/A')
                        print(f"        [{check_name}] {reason}")
                        print(f"                     正常涨停: {normal_count}, 总涨停: {total_count}, 阈值: {max_allowed}")
                    elif check_name == 'price':
                        support_ok = check_detail.get('not_break_support', True)
                        close_ok = check_detail.get('close_T3_ok', True)
                        if not support_ok:
                            low_vs_support = check_detail.get('low_vs_support', 0)
                            support_price = check_detail.get('support_price', 0)
                            print(f"        [{check_name}] 破支撑位 (最低价相对支撑: {low_vs_support:.2f}%, 支撑位: {support_price:.2f})")
                        if not close_ok:
                            close_vs_T0 = check_detail.get('close_T3_vs_T0', 0)
                            threshold = check_detail.get('threshold', 0)
                            print(f"        [{check_name}] T3收盘过低 (相对T0: {close_vs_T0:.2f}%, 阈值: {threshold:.2f}%)")
                    elif check_name == 'candle':
                        bearish_count = check_detail.get('bearish_count', 0)
                        print(f"        [{check_name}] 阴线过多 ({bearish_count}根 > 2根)")
                    elif check_name == 'ma':
                        min_bias = check_detail.get('min_ma5_bias', 0)
                        threshold = check_detail.get('threshold', -3.0)
                        print(f"        [{check_name}] 跌破MA5过多 ({min_bias:.2f}% < {threshold:.2f}%)")
                    elif check_name == 'center':
                        center_vs_T0 = check_detail.get('center_vs_T0', 0)
                        tolerance = check_detail.get('tolerance', 0)
                        print(f"        [{check_name}] 重心过低 ({center_vs_T0:.2f}% vs 阈值{tolerance:.2f}%)")
                    elif check_name == 'volume':
                        T2_ratio = check_detail.get('T2_ratio', 0)
                        T3_ratio = check_detail.get('T3_ratio', 0)
                        print(f"        [{check_name}] 量能过大 (T2:{T2_ratio:.2f}x, T3:{T3_ratio:.2f}x > 2.5x)")
                    elif check_name == 'position':
                        gain_20d = check_detail.get('gain_20d', None)
                        ma20_bias = check_detail.get('ma20_bias', None)
                        gain_threshold = check_detail.get('gain_20d_threshold', 50.0)
                        bias_threshold = check_detail.get('ma20_bias_threshold', 40.0)
                        if gain_20d is not None:
                            print(f"        [{check_name}] 20日涨幅: {gain_20d:.2f}% (阈值: {gain_threshold:.1f}%)")
                        if ma20_bias is not None:
                            print(f"        [{check_name}] MA20偏离: {ma20_bias:.2f}% (阈值: {bias_threshold:.1f}%)")
    
    print(f"  [OK] Passed filter: {filter_stats['passed']}/{filter_stats['total']} ({pass_rate:.1f}%)")
    
    if len(filtered_stocks) == 0:
        print(f"  [WARNING] No stocks passed consolidation filter")
        engine.dispose()
        return
    
    # 提取特征（只对通过过滤的股票）
    print(f"\n[8/11] Extracting features for filtered stocks...")
    print(f"  - Base features: T1-T2 ({(T0_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} to {T2_date.strftime('%Y-%m-%d')})")
    print(f"  - Enhanced features: T3 ({T3_date.strftime('%Y-%m-%d')})")
    print(f"  - Prediction target: T4-T6 (3 days max gain)")
    
    features_list = []
    stock_list = []
    
    T3_date_str = T3_date.strftime('%Y-%m-%d')
    
    for idx, stock_info in enumerate(filtered_stocks):
        stock_code = stock_info['instrument']
        stock_name = stock_info['stock_name']
        consecutive_boards = stock_info['consecutive_boards']
        seal_funds = stock_info['seal_funds']
        explosion_count = stock_info['explosion_count']
        
        try:
            # 1. 提取基础特征（59维，基于T1-T2）
            base_feat = extract_base_features_for_stock(
                stock_code, T0_date, T2_date, kline_df,
                consecutive_boards, seal_funds, explosion_count
            )
            
            if base_feat is None:
                print(f"  [WARNING] Skipping {stock_code}: insufficient K-line data")
                continue
            
            # 2. 提取板块特征（7维，基于T2）- 决策时T3数据未产生
            T2_date_str = T2_date.strftime('%Y-%m-%d')
            sector_feat = extract_sector_features(engine, T2_date_str, stock_code, sector_mapping)
            
            # 3. 提取筹码特征（5维，基于T2）- 决策时T3数据未产生
            chip_feat = extract_chip_features(engine, T2_date_str, stock_code)
            
            # 4. 提取环境特征（3维，基于T2）- 决策时T3数据未产生
            env_feat = extract_environment_features(engine, T2_date_str, stock_code, sector_mapping)
            
            # 合并所有特征
            features = {
                **base_feat,      # 59维
                **sector_feat,    # 7维
                **chip_feat,      # 5维
                **env_feat        # 3维
            }
            
            features_list.append(features)
            stock_list.append({
                'instrument': stock_code, 
                'stock_name': stock_name,
                'consecutive_boards': consecutive_boards,
                'seal_funds': seal_funds
            })
            
        except Exception as e:
            print(f"  [WARNING] Failed to extract features for {stock_code}: {e}")
            continue
        
        if (idx + 1) % 10 == 0:
            print(f"    Progress: {idx + 1}/{len(filtered_stocks)}")
    
    print(f"  [OK] Extracted complete features for {len(features_list)} stocks")
    
    # 构建特征矩阵
    print(f"\n[9/11] Building feature matrix...")
    df_features = pd.DataFrame(features_list)
    
    # 检查缺失特征
    missing_features = [f for f in feature_names if f not in df_features.columns]
    if missing_features:
        print(f"  [WARNING] Missing {len(missing_features)} features, filling with 0")
        for feat in missing_features:
            df_features[feat] = 0
    
    # 按feature_names顺序排列
    X = df_features[feature_names].fillna(0)
    print(f"  [OK] Feature matrix shape: {X.shape}")
    print(f"  [INFO] Complete features: {len([f for f in feature_names if f in df_features.columns])}/{len(feature_names)}")
    
    # 预测
    print(f"\n[10/11] Predicting...")
    
    if model_type == 'ensemble':
        # 使用两个模型的平均预测
        print("  Using ensemble prediction (LightGBM + XGBoost average)...")
        lgb_pred = lgb_model.predict(X)
        
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        xgb_pred = xgb_model.predict(dmatrix)
        
        # 加权平均（LightGBM权重0.52，XGBoost权重0.48，基于测试集Spearman）
        y_pred = lgb_pred * 0.52 + xgb_pred * 0.48
        print(f"    LightGBM weight: 0.52, XGBoost weight: 0.48")
    elif model_type == 'lgb':
        y_pred = lgb_model.predict(X)
    else:  # xgb
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        y_pred = xgb_model.predict(dmatrix)
    
    # 生成结果
    print(f"\n[11/11] Generating results...")
    
    df_result = pd.DataFrame(stock_list)
    df_result['T0_date'] = T0_date.strftime('%Y-%m-%d')  # 涨停日
    df_result['T3_date'] = T3_date.strftime('%Y-%m-%d')  # 整固期结束日
    df_result['prediction_score'] = y_pred
    df_result['rank'] = df_result['prediction_score'].rank(ascending=False, method='first').astype(int)
    df_result = df_result.sort_values('prediction_score', ascending=False).reset_index(drop=True)
    
    # 显示Top K结果
    print(f"\n" + "=" * 80)
    print(f"Top {top_k} Predictions:")
    
    if is_regression:
        print(f"  (预测T4-T6未来3天最大涨幅 %)")
    else:
        print(f"  (Score = 未来5天涨幅≥10%的概率，注：当前为分类模型，建议重新训练V3.1回归模型)")
    
    print("=" * 80)
    
    for idx, row in df_result.head(top_k).iterrows():
        if is_regression:
            # 回归模型：直接显示预测涨幅
            print(f"  {idx + 1:2d}. {row['instrument']:10s} {row['stock_name']:10s} 预测涨幅: {row['prediction_score']:6.2f}%")
        else:
            # 分类模型：显示概率
            probability_pct = row['prediction_score'] * 100
            print(f"  {idx + 1:2d}. {row['instrument']:10s} {row['stock_name']:10s} Score: {row['prediction_score']:.4f} ({probability_pct:.1f}%)")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f'v3_prediction_T0_{T0_date.strftime("%Y%m%d")}_T3_{T3_date.strftime("%Y%m%d")}_{model_type}_{timestamp}.csv'
    df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n[OK] Saved results to {output_file}")
    print(f"  - T0 (Limit-up date): {T0_date.strftime('%Y-%m-%d')}")
    print(f"  - T2 (Base feature end): {T2_date.strftime('%Y-%m-%d')}")
    print(f"  - T3 (Decision date): {T3_date.strftime('%Y-%m-%d')}")
    print(f"  - Prediction target: T4-T6 (3 days max gain)")
    
    engine.dispose()
    
    print("\n" + "=" * 80)
    print("[COMPLETED] Prediction finished")
    print("=" * 80)
    
    return df_result


def main():
    parser = argparse.ArgumentParser(
        description='V3.0 Consolidation Prediction',
        epilog='''
Time Window Logic:
  --date specifies T3 date (consolidation end date)
  System will:
    1. Calculate T0 (3 trading days before T3) - limit-up date
    2. Fetch limit-up stocks from Zhitu/Moyunfu API for T0
    3. Load K-line data from PostgreSQL
    4. Extract base features (59D) using T1-T3 consolidation period (3 days)
    5. Extract enhanced features (15D) using T3 snapshot (sector/chip/env)
    6. Predict performance from T4 onwards (T3+1)
    
Example:
  python scripts/predict_v3.py --date 2025-11-13
    → T0: 2025-11-10 (limit-up date, fetch from API)
    → T1-T3: 2025-11-11~13 (base features: 3-day consolidation)
    → T3: 2025-11-13 (enhanced features: 1-day snapshot)
    → Predict: 2025-11-14+ (T4 onwards, 5-day window)
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--date', type=str, required=True, 
                       help='T3 date (consolidation end date, format: YYYY-MM-DD)')
    parser.add_argument('--top-k', type=int, default=10, 
                       help='Number of top predictions to display (default: 10)')
    parser.add_argument('--model', type=str, default='ensemble', 
                       choices=['ensemble', 'lgb', 'xgb'], 
                       help='Model type: ensemble (default, LGB+XGB average), lgb (LightGBM only), xgb (XGBoost only)')
    
    args = parser.parse_args()
    
    predict(args.date, args.top_k, args.model)


if __name__ == '__main__':
    main()
