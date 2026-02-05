#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强势整理策略 - 特征提取模块

功能：
提取7大类特征：
1. 形态特征（21个）- T+1~T+3的K线形态
2. 均线特征（12个）- MA偏离度、多头排列等
3. 量能特征（8个）- 量比、缩量率等
4. 股性特征（8个）- 从股性文件直接获取
5. 市场特征（0个）- 已移除（V2.3）
6. 前期表现特征（4个）- 前3日详细表现（V2.4.1精简）
7. 相对强度特征（2个）- 个股vs市场（V2.4.1精简）

总特征数：53（V2.3）→ 63（V2.4）→ 59（V2.4.1）

更新历史：
- V2.3 (2026-01-28): 移除5个市场情绪特征，Spearman从0.078提升到0.108 (+39%)
- V2.4 (2026-01-28): 添加10个新特征，Spearman=0.103（未达预期，下降5%）
- V2.4.1 (2026-01-28): 移除4个无用/重复特征，保留6个优质新特征，目标Spearman>0.12

移除特征（V2.4.1）：
- prior_max_drawdown（相关性-0.027，几乎无关）
- prior_shadow_symmetry（相关性0.022，几乎无关）
- relative_amplitude_vs_market（与prior_total_amplitude重复，相关性0.097）
- relative_volume_ratio_vs_market（相关性-0.035，负相关）

保留新特征（V2.4.1）：
- relative_gain_vs_market（相关性0.106，最佳新特征！）
- prior_total_amplitude（相关性0.097，优秀）
- prior_price_stability（相关性0.085，良好）
- relative_turnover_vs_market（相关性0.062，一般）
- prior_total_turnover（相关性0.062，一般）
- prior_bullish_ratio（相关性0.060，一般）

作者：AI Assistant
日期：2026-01-28
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging


class ConsolidationFeatureExtractor:
    """特征提取器"""
    
    @staticmethod
    def extract_pattern_features(
        kline_ndays: pd.DataFrame,
        T0_close: float
    ) -> Dict[str, float]:
        """
        提取形态特征（24个）
        
        V3.2版本（新增3个整理期涨停特征）：
        - consolidation_limit_up_count: 整理期涨停次数
        - consolidation_has_consecutive_limit_up: 是否连续涨停
        - consolidation_limit_up_turnover_avg: 涨停日平均换手率
        
        V3.1版本：基于T+1~T+2的K线数据（2天）
        V3.0版本：基于T+1~T+3的K线数据（3天）
        
        Args:
            kline_ndays: T0~T+N的K线数据（N=2或3）
            T0_close: T0涨停价
        
        Returns:
            形态特征字典（24维）
        """
        # 自动适配：如果传入3天（T0,T1,T2），取T1,T2；如果传入4天（T0,T1,T2,T3），取T1,T2,T3
        n_days = len(kline_ndays)
        if n_days == 3:
            # V3.1: T0, T1, T2 → 使用 T1, T2
            kline_target = kline_ndays.iloc[1:3].copy()  # T1, T2 (使用.copy()避免pandas视图问题)
        elif n_days == 4:
            # V3.0: T0, T1, T2, T3 → 使用 T1, T2, T3
            kline_target = kline_ndays.iloc[1:4].copy()  # T1, T2, T3 (使用.copy()避免pandas视图问题)
        else:
            raise ValueError(f"Expected 3 or 4 days of kline data, got {n_days}")
        
        target_days = len(kline_target)
        
        # DEBUG: 检查kline_target长度
        if target_days < 2:
            raise ValueError(f"kline_target too short: {target_days}, expected >= 2")
        
        highs = kline_target['high'].values
        lows = kline_target['low'].values
        opens = kline_target['open'].values
        closes = kline_target['close'].values
        
        features = {}
        
        # 1. 最低价斜率（低点抬升趋势）
        # 计算N天最低价的线性回归斜率（N=2或3）
        x = np.arange(1, target_days + 1)  # [1,2] 或 [1,2,3]
        slope_low = np.polyfit(x, lows, 1)[0]
        features['low_slope'] = slope_low / T0_close  # 归一化
        
        # 2. 收盘价重心相对T0
        # N天收盘价的平均值相对T0涨停价的位置
        avg_close = closes.mean()
        features['close_center_vs_T0'] = avg_close / T0_close
        
        # 3. 平均收盘价位置
        # 收盘价在当日高低价区间的平均位置
        close_positions = [(closes[i] - lows[i]) / (highs[i] - lows[i] + 1e-9) 
                          for i in range(len(closes))]
        features['avg_close_position'] = np.mean(close_positions)
        
        # 4. 平均实体比例
        # K线实体占全天振幅的平均比例
        body_ratios = [abs(closes[i] - opens[i]) / (highs[i] - lows[i] + 1e-9) 
                      for i in range(len(closes))]
        features['avg_body_ratio'] = np.mean(body_ratios)
        
        # 5. 阳线数量
        bullish_count = sum([1 for i in range(len(closes)) if closes[i] >= opens[i]])
        features['bullish_count'] = bullish_count
        
        # 6. 平均振幅
        amplitudes = [(highs[i] - lows[i]) / opens[i] for i in range(len(opens))]
        features['amplitude_avg'] = np.mean(amplitudes)
        
        # 7. 最大振幅
        features['amplitude_max'] = max(amplitudes)
        
        # 8. 最小振幅
        features['amplitude_min'] = min(amplitudes)
        
        # 9. 振幅标准差
        features['amplitude_std'] = np.std(amplitudes)
        
        # 10. 上影线平均长度
        upper_shadows = [(highs[i] - max(opens[i], closes[i])) / (highs[i] - lows[i] + 1e-9) 
                        for i in range(len(highs))]
        features['upper_shadow_avg'] = np.mean(upper_shadows)
        
        # 11. 下影线平均长度
        lower_shadows = [(min(opens[i], closes[i]) - lows[i]) / (highs[i] - lows[i] + 1e-9) 
                        for i in range(len(lows))]
        features['lower_shadow_avg'] = np.mean(lower_shadows)
        
        # 12. 最高价斜率
        slope_high = np.polyfit(x, highs, 1)[0]
        features['high_slope'] = slope_high / T0_close
        
        # 13. 收盘价斜率
        slope_close = np.polyfit(x, closes, 1)[0]
        features['close_slope'] = slope_close / T0_close
        
        # 14. 价格区间收敛度
        # 最后一天的振幅相对第一天的振幅
        features['range_convergence'] = amplitudes[-1] / (amplitudes[0] + 1e-9)
        
        # 15. T+1相对T0的涨幅
        features['T1_gain'] = (closes[0] - T0_close) / T0_close
        
        # 16. T+2相对T+1的涨幅
        features['T2_gain'] = (closes[1] - closes[0]) / closes[0]
        
        # 17. T+3相对T+2的涨幅（如果有T3的话）
        if target_days >= 3:
            features['T3_gain'] = (closes[2] - closes[1]) / closes[1]
        else:
            features['T3_gain'] = 0.0  # V3.1没有T3，用0填充
        
        # 18. 累计涨幅（最后一天相对T0）
        features['cumulative_gain'] = (closes[-1] - T0_close) / T0_close
        
        # 19. 最大回撤（相对T0）
        min_close = closes.min()
        features['max_drawdown'] = (min_close - T0_close) / T0_close
        
        # 20. 价格波动率
        features['price_volatility'] = closes.std() / closes.mean()
        
        # 21. 整理期涨停次数（新增特征）
        # 统计 T1-T2（或T1-T2-T3）期间的涨停次数
        # 涨停判断：涨幅 >= 9.5%
        limit_up_count = 0
        limit_up_days = []
        turns = kline_target.get('turn', pd.Series([0] * len(kline_target))).values  # 获取换手率，默认0
        
        # 计算每天相对前一天的涨幅（需要加上T0数据来计算T1涨幅）
        prev_close = T0_close
        for i in range(len(closes)):
            gain_pct = (closes[i] - prev_close) / prev_close * 100
            if gain_pct >= 9.5:  # 涨停阈值
                limit_up_count += 1
                limit_up_days.append(i)
            prev_close = closes[i]
        
        features['consolidation_limit_up_count'] = limit_up_count
        
        # 22. 是否连续涨停（新增特征）
        # 检查涨停日是否连续（如[0,1]表示T1,T2连续涨停）
        is_consecutive = 0
        if len(limit_up_days) >= 2:
            # 检查是否连续
            is_consecutive = all(limit_up_days[i+1] - limit_up_days[i] == 1 
                               for i in range(len(limit_up_days)-1))
        features['consolidation_has_consecutive_limit_up'] = int(is_consecutive)
        
        # 23. 整理期涨停日平均换手率（新增特征）
        # 如果有涨停，计算这些涨停日的平均换手率
        if limit_up_count > 0 and turns is not None:
            limit_up_turnover = [turns[i] for i in limit_up_days]
            features['consolidation_limit_up_turnover_avg'] = np.mean(limit_up_turnover)
        else:
            features['consolidation_limit_up_turnover_avg'] = 0.0
        
        # 24. 形态综合评分（更新编号）
        # 综合多个指标，评估形态强弱
        pattern_score = (
            (features['low_slope'] > 0) * 0.15 +  # 低点抬升
            (features['close_center_vs_T0'] >= 1.0) * 0.15 +  # 重心在T0之上
            (features['avg_close_position'] > 0.5) * 0.15 +  # 收盘价偏上
            (features['bullish_count'] >= 2) * 0.15 +  # 多数阳线
            (features['cumulative_gain'] >= 0) * 0.15 +  # 累计上涨
            (features['consolidation_limit_up_count'] >= 1) * 0.25  # 整理期有涨停（新增权重）
        )
        features['pattern_score'] = pattern_score
        
        return features
    
    @staticmethod
    def extract_ma_features(
        kline_df: pd.DataFrame,
        T0_idx: int,
        target_day: int = 3
    ) -> Dict[str, float]:
        """
        提取均线特征（12个）
        
        V3.1版本：支持指定目标天数（T2或T3）
        
        Args:
            kline_df: 完整K线数据（包含MA）
            T0_idx: T0在kline_df中的索引位置
            target_day: 目标天数（V3.1: 2=T2, V3.0: 3=T3）
        
        Returns:
            均线特征字典
        """
        # 检查是否有足够的数据
        if T0_idx + target_day + 1 > len(kline_df):
            return {}
        
        # 获取目标天的数据（V3.1: T2, V3.0: T3）
        target_data = kline_df.iloc[T0_idx + target_day]
        target_close = target_data['close']
        target_ma5 = target_data.get('ma5', np.nan)
        target_ma10 = target_data.get('ma10', np.nan)
        target_ma20 = target_data.get('ma20', np.nan)
        
        features = {}
        
        # 1-3. MA偏离度
        if not pd.isna(target_ma5):
            features['ma5_bias'] = (target_close - target_ma5) / target_ma5
        else:
            features['ma5_bias'] = 0.0
            
        if not pd.isna(target_ma10):
            features['ma10_bias'] = (target_close - target_ma10) / target_ma10
        else:
            features['ma10_bias'] = 0.0
            
        if not pd.isna(target_ma20):
            features['ma20_bias'] = (target_close - target_ma20) / target_ma20
        else:
            features['ma20_bias'] = 0.0
        
        # 4. 多头排列度
        # MA5 > MA10 > MA20 的程度
        multi_bull_degree = 0
        if not pd.isna(target_ma5) and not pd.isna(target_ma10) and target_ma5 > target_ma10:
            multi_bull_degree += 0.5
        if not pd.isna(target_ma10) and not pd.isna(target_ma20) and target_ma10 > target_ma20:
            multi_bull_degree += 0.5
        features['multi_bull_degree'] = multi_bull_degree
        
        # 5-7. MA均线散度
        if not pd.isna(target_ma5) and not pd.isna(target_ma10):
            features['ma5_ma10_divergence'] = (target_ma5 - target_ma10) / target_ma10
        else:
            features['ma5_ma10_divergence'] = 0.0
            
        if not pd.isna(target_ma5) and not pd.isna(target_ma20):
            features['ma5_ma20_divergence'] = (target_ma5 - target_ma20) / target_ma20
        else:
            features['ma5_ma20_divergence'] = 0.0
            
        if not pd.isna(target_ma10) and not pd.isna(target_ma20):
            features['ma10_ma20_divergence'] = (target_ma10 - target_ma20) / target_ma20
        else:
            features['ma10_ma20_divergence'] = 0.0
        
        # 8-10. T+1到目标天期间在MA5之上的天数（V3.1: T1-T2, V3.0: T1-T3）
        kline_period = kline_df.iloc[T0_idx+1:T0_idx+target_day+1]
        days_above_ma5 = 0
        days_above_ma10 = 0
        days_above_ma20 = 0
        
        for _, row in kline_period.iterrows():
            if not pd.isna(row.get('ma5')) and row['close'] >= row['ma5']:
                days_above_ma5 += 1
            if not pd.isna(row.get('ma10')) and row['close'] >= row['ma10']:
                days_above_ma10 += 1
            if not pd.isna(row.get('ma20')) and row['close'] >= row['ma20']:
                days_above_ma20 += 1
        
        # 归一化（除以实际天数）
        period_days = target_day  # V3.1: 2天, V3.0: 3天
        features['days_above_ma5'] = days_above_ma5 / period_days
        features['days_above_ma10'] = days_above_ma10 / period_days
        features['days_above_ma20'] = days_above_ma20 / period_days
        
        # 11. MA5斜率（目标天相对T0的MA5变化）
        T0_ma5 = kline_df.iloc[T0_idx].get('ma5', np.nan)
        if not pd.isna(T0_ma5) and not pd.isna(target_ma5):
            features['ma5_slope'] = (target_ma5 - T0_ma5) / T0_ma5
        else:
            features['ma5_slope'] = 0.0
        
        # 12. 价格相对MA5的平均位置
        ma5_positions = []
        for _, row in kline_period.iterrows():
            if not pd.isna(row.get('ma5')):
                ma5_positions.append((row['close'] - row['ma5']) / row['ma5'])
        
        if ma5_positions:
            features['avg_ma5_position'] = np.mean(ma5_positions)
        else:
            features['avg_ma5_position'] = 0.0
        
        return features
    
    @staticmethod
    def extract_volume_features(
        kline_4days: pd.DataFrame,
        stock_avg_volume: float
    ) -> Dict[str, float]:
        """
        提取量能特征（8个）
        
        V3.1版本：支持3天或4天K线数据
        
        Args:
            kline_4days: T0~T+N的K线数据（N=2或3，即3天或4天）
            stock_avg_volume: 该股票的20日平均成交量
        
        Returns:
            量能特征字典
        """
        # 自动适配：3天取T1-T2，4天取T1-T3
        n_days = len(kline_4days)
        if n_days == 3:
            kline_target = kline_4days.iloc[1:3].copy()  # T1, T2
        elif n_days == 4:
            kline_target = kline_4days.iloc[1:4].copy()  # T1, T2, T3
        else:
            raise ValueError(f"Expected 3 or 4 days of kline data, got {n_days}")
        
        T0_volume = kline_4days.iloc[0]['volume']
        
        volumes = kline_target['volume'].values
        
        features = {}
        
        # 1. 平均量比
        avg_volume_ratio = volumes.mean() / stock_avg_volume
        features['avg_volume_ratio'] = avg_volume_ratio
        
        # 2. 量能衰减率
        # 从第1天到最后1天的量能变化趋势
        if volumes[0] > 0 and len(volumes) >= 2:
            volume_decay = (volumes[-1] - volumes[0]) / volumes[0]
        else:
            volume_decay = 0.0
        features['volume_decay_rate'] = volume_decay
        
        # 3. 总换手率（如果数据中有turn字段）
        if 'turn' in kline_target.columns:
            total_turnover = kline_target['turn'].sum()
        else:
            # 用量比近似估算
            total_turnover = avg_volume_ratio * len(kline_target) * 5  # 假设日均换手5%
        features['total_turnover'] = total_turnover
        
        # 4. 量比相对T0
        T1_volume_vs_T0 = volumes[0] / (T0_volume + 1e-9)
        features['volume_vs_T0'] = T1_volume_vs_T0
        
        # 5. 量价相关性（添加空值和标准差检查，避免除零警告）
        closes = kline_target['close'].values
        if len(volumes) >= 3 and len(closes) >= 3:
            # 检查是否有NaN或标准差为0（避免除零警告）
            if (not np.any(np.isnan(volumes)) and not np.any(np.isnan(closes)) and
                np.std(volumes) > 1e-9 and np.std(closes) > 1e-9):
                try:
                    corr = np.corrcoef(volumes, closes)[0, 1]
                    if not np.isnan(corr):
                        features['price_volume_corr'] = corr
                    else:
                        features['price_volume_corr'] = 0.0
                except Exception:
                    # 兜底：任何异常都返回0
                    features['price_volume_corr'] = 0.0
            else:
                # 标准差为0或存在NaN，无法计算相关性
                features['price_volume_corr'] = 0.0
        else:
            features['price_volume_corr'] = 0.0
        
        # 6. 量能波动率
        features['volume_volatility'] = volumes.std() / (volumes.mean() + 1e-9)
        
        # 7. 最大量比
        features['max_volume_ratio'] = volumes.max() / stock_avg_volume
        
        # 8. 最小量比
        features['min_volume_ratio'] = volumes.min() / stock_avg_volume
        
        return features
    
    @staticmethod
    def extract_stock_character_features(
        instrument: str,
        T3_date: pd.Timestamp,
        stock_char_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        提取股性特征（7个）
        
        从预计算的股性特征文件中获取
        
        Args:
            instrument: 股票代码
            T3_date: T+3日期
            stock_char_df: 股性特征数据
        
        Returns:
            股性特征字典
        """
        # 获取T+3日的股性特征
        stock_char = stock_char_df[
            (stock_char_df['instrument'] == instrument) &
            (stock_char_df['date'] == T3_date)
        ]
        
        features = {}
        
        if len(stock_char) == 0:
            # 如果没有找到，返回默认值
            features['volatility_60d'] = 0.0
            features['up_day_ratio_60d'] = 0.5
            features['rebound_speed'] = 10.0
            features['amplitude_avg_60d'] = 0.03
            features['up_body_sum_ratio_60d'] = 1.0
            features['volume_price_correlation'] = 0.0
            features['turnover_rate'] = 5.0
        else:
            row = stock_char.iloc[0]
            # 提取关键股性特征
            features['volatility_60d'] = row.get('volatility_60d', 0.0)
            features['up_day_ratio_60d'] = row.get('up_day_ratio_60d', 0.5)
            features['rebound_speed'] = row.get('rebound_speed', 10.0)
            features['amplitude_avg_60d'] = row.get('amplitude_avg_60d', 0.03)
            features['up_body_sum_ratio_60d'] = row.get('up_body_sum_ratio_60d', 1.0)
            features['volume_price_correlation'] = row.get('volume_price_correlation', 0.0)
            features['turnover_rate'] = row.get('turnover_rate', 5.0)
        
        # 处理缺失值
        for key in features:
            if pd.isna(features[key]):
                if key == 'up_day_ratio_60d':
                    features[key] = 0.5
                elif key == 'rebound_speed':
                    features[key] = 10.0
                elif key == 'turnover_rate':
                    features[key] = 5.0
                else:
                    features[key] = 0.0
        
        return features
    
    @staticmethod
    def extract_market_features(
        T0_date: pd.Timestamp,
        T3_date: pd.Timestamp,
        sentiment_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        提取市场特征（5个）
        
        Args:
            T0_date: T0日期
            T3_date: T+3日期
            sentiment_df: 市场情绪数据
        
        Returns:
            市场特征字典
        """
        features = {}
        
        # ❌ 市场情绪特征已被禁用（2026-01-28诊断发现这些特征严重干扰模型）
        # 诊断结果显示：这5个特征占据模型重要性前60%，但与收益相关性极低（<0.08）
        # 导致模型过拟合市场整体走势，而忽略个股自身特征
        # 详见：docs/optimization/MODEL_FAILURE_ROOT_CAUSE_ANALYSIS.md
        
        # # 获取T0~T+3期间的市场情绪数据
        # period_sentiment = sentiment_df[
        #     (sentiment_df['crawl_date'] >= T0_date) &
        #     (sentiment_df['crawl_date'] <= T3_date)
        # ]
        # 
        # if len(period_sentiment) == 0:
        #     # 如果没有数据，返回中性值
        #     features['market_activity_avg'] = 50.0
        #     features['limit_up_count_avg'] = 50.0
        #     features['sentiment_score_avg'] = 50.0
        #     features['median_gain_avg'] = 0.0
        #     features['market_trend'] = 0.0
        # else:
        #     # 1. 平均市场活跃度
        #     if 'market_activity' in period_sentiment.columns:
        #         features['market_activity_avg'] = period_sentiment['market_activity'].mean()
        #     else:
        #         features['market_activity_avg'] = 50.0
        #     
        #     # 2. 平均涨停数量
        #     if 'limit_up_count' in period_sentiment.columns:
        #         features['limit_up_count_avg'] = period_sentiment['limit_up_count'].mean()
        #     else:
        #         features['limit_up_count_avg'] = 50.0
        #     
        #     # 3. 平均情绪分数
        #     if 'sentiment_score' in period_sentiment.columns:
        #         features['sentiment_score_avg'] = period_sentiment['sentiment_score'].mean()
        #     else:
        #         features['sentiment_score_avg'] = 50.0
        #     
        #     # 4. 平均涨幅中位数
        #     if 'median_gain_all' in period_sentiment.columns:
        #         features['median_gain_avg'] = period_sentiment['median_gain_all'].mean()
        #     else:
        #         features['median_gain_avg'] = 0.0
        #     
        #     # 5. 市场趋势（T+3相对T0的变化）
        #     if len(period_sentiment) >= 2:
        #         first_sentiment = period_sentiment.iloc[0].get('sentiment_score', 50.0)
        #         last_sentiment = period_sentiment.iloc[-1].get('sentiment_score', 50.0)
        #         if not pd.isna(first_sentiment) and not pd.isna(last_sentiment) and first_sentiment != 0:
        #             features['market_trend'] = (last_sentiment - first_sentiment) / first_sentiment
        #         else:
        #             features['market_trend'] = 0.0
        #     else:
        #         features['market_trend'] = 0.0
        
        return features
    
    @staticmethod
    def extract_prior_performance_features(
        kline_4days: pd.DataFrame,
        T0_close: float
    ) -> Dict[str, float]:
        """
        提取前期表现特征
        
        V3.1版本：支持T1-T2（2天）或T1-T3（3天）
        
        这些特征捕捉整理阶段的质量：
        - 回撤控制能力
        - 换手率累积（活跃度）
        - 振幅累积（波动性）
        - 上下影线对称性（支撑/压力）
        
        Args:
            kline_4days: T0~T+N的K线数据（N=2或3，即3天或4天）
            T0_close: T0涨停价
        
        Returns:
            前期表现特征字典
        """
        # 自动适配：3天取T1-T2，4天取T1-T3
        n_days = len(kline_4days)
        if n_days == 3:
            kline_target = kline_4days.iloc[1:3].copy()  # T1, T2
        elif n_days == 4:
            kline_target = kline_4days.iloc[1:4].copy()  # T1, T2, T3
        else:
            raise ValueError(f"Expected 3 or 4 days of kline data, got {n_days}")
        
        highs = kline_target['high'].values
        lows = kline_target['low'].values
        opens = kline_target['open'].values
        closes = kline_target['close'].values
        
        features = {}
        
        # ❌ 1. 前3日最大回撤 - 已移除（V2.4.1）
        # 诊断发现：相关性仅-0.027，与收益几乎无关
        # min_price = lows.min()
        # max_drawdown_from_T0 = (T0_close - min_price) / T0_close
        # features['prior_max_drawdown'] = max_drawdown_from_T0
        
        # 2. 前N日换手率总和（N=2或3）
        # 衡量整理期间的活跃度
        if 'turn' in kline_target.columns or 'turnover' in kline_target.columns:
            turnover_col = 'turn' if 'turn' in kline_target.columns else 'turnover'
            total_turnover = kline_target[turnover_col].sum()
        else:
            # 如果没有换手率数据，使用量比估算
            total_turnover = 0.0
        features['prior_total_turnover'] = total_turnover
        
        # 3. 前N日振幅总和（N=2或3）
        # 衡量整理期间的总体波动
        amplitudes = [(highs[i] - lows[i]) / opens[i] for i in range(len(opens))]
        total_amplitude = sum(amplitudes)
        features['prior_total_amplitude'] = total_amplitude
        
        # ❌ 4. 上下影线对称性 - 已移除（V2.4.1）
        # 诊断发现：相关性仅0.022，与收益几乎无关
        # upper_shadows = [(highs[i] - max(opens[i], closes[i])) / (highs[i] - lows[i] + 1e-9) 
        #                 for i in range(len(highs))]
        # lower_shadows = [(min(opens[i], closes[i]) - lows[i]) / (highs[i] - lows[i] + 1e-9) 
        #                 for i in range(len(lows))]
        # 
        # avg_upper_shadow = np.mean(upper_shadows)
        # avg_lower_shadow = np.mean(lower_shadows)
        # 
        # if avg_lower_shadow > 0:
        #     shadow_ratio = avg_upper_shadow / (avg_lower_shadow + 1e-9)
        # else:
        #     shadow_ratio = 1.0
        # features['prior_shadow_symmetry'] = shadow_ratio
        
        # 5. 价格重心稳定性
        # 衡量3日收盘价的标准差（越小越稳定）
        close_std = np.std(closes)
        close_stability = close_std / T0_close  # 归一化
        features['prior_price_stability'] = close_stability
        
        # 6. 连续阳线比例
        # 衡量整理期间的多头力量
        bullish_days = sum([1 for i in range(len(closes)) if closes[i] >= opens[i]])
        bullish_ratio = bullish_days / 3.0
        features['prior_bullish_ratio'] = bullish_ratio
        
        return features
    
    @staticmethod
    def extract_relative_strength_features(
        stock_kline: pd.DataFrame,
        T0_idx: int,
        index_kline_df: Optional[pd.DataFrame] = None,
        n_days: int = 3
    ) -> Dict[str, float]:
        """
        提取相对强度特征（个股 vs 大盘/市场）
        
        V3.1版本：支持2天或3天整理期
        
        这些特征衡量个股相对于市场的表现：
        - 个股收益 vs 大盘收益（超额收益）
        - 个股换手 vs 市场平均换手（相对活跃度）
        - 个股振幅 vs 市场平均振幅（相对波动性）
        
        Args:
            stock_kline: 个股K线数据
            T0_idx: T0在数据中的位置
            index_kline_df: 指数K线数据（可选，如果没有则计算全市场平均）
            n_days: 整理期天数（V3.1: 2天，V3.0: 3天）
        
        Returns:
            相对强度特征字典
        """
        features = {}
        
        # 检查是否有足够的数据（T0 + 1 到 T0 + n_days）
        if T0_idx + n_days + 1 > len(stock_kline):
            return {
                'relative_gain_vs_market': 0.0,
                'relative_turnover_vs_market': 1.0,
                # ❌ 以下已移除（V2.4.1）
                # 'relative_amplitude_vs_market': 1.0,
                # 'relative_volume_ratio_vs_market': 1.0
            }
        
        # 获取T+1到T+n_days的数据（V3.1: T1-T2，V3.0: T1-T3）
        kline_target = stock_kline.iloc[T0_idx+1:T0_idx+n_days+1].copy()
        T0_close = stock_kline.iloc[T0_idx]['close']
        
        # 1. 个股N日累计收益
        Tn_close = kline_target.iloc[-1]['close']
        stock_gain = (Tn_close - T0_close) / T0_close
        
        # 2. 个股平均换手率
        if 'turn' in kline_target.columns or 'turnover' in kline_target.columns:
            turnover_col = 'turn' if 'turn' in kline_target.columns else 'turnover'
            stock_avg_turnover = kline_target[turnover_col].mean()
        else:
            stock_avg_turnover = 0.05  # 默认5%
        
        # 3. 个股平均振幅
        stock_amplitudes = [(row['high'] - row['low']) / row['open'] 
                           for _, row in kline_target.iterrows()]
        stock_avg_amplitude = np.mean(stock_amplitudes)
        
        # 4. 个股平均量比
        if T0_idx >= 20:
            stock_avg_volume_20d = stock_kline['volume'].iloc[T0_idx-20:T0_idx].mean()
        else:
            stock_avg_volume_20d = stock_kline['volume'].iloc[:T0_idx].mean()
        
        if stock_avg_volume_20d > 0:
            stock_avg_volume_ratio = kline_target['volume'].mean() / stock_avg_volume_20d
        else:
            stock_avg_volume_ratio = 1.0
        
        # 如果有指数数据，计算相对指数；否则使用固定市场基准
        if index_kline_df is not None and len(index_kline_df) > 0:
            # TODO: 实现相对指数的计算
            # 目前暂时使用市场平均基准
            market_gain = 0.0  # 市场N日平均涨幅（待实现）
            market_avg_turnover = 0.05  # 市场平均换手率
            market_avg_amplitude = 0.03  # 市场平均振幅
            market_avg_volume_ratio = 1.0  # 市场平均量比
        else:
            # 使用经验基准值
            market_gain = 0.0  # 假设市场平均涨幅为0
            market_avg_turnover = 0.05  # 假设市场平均换手率5%
            market_avg_amplitude = 0.03  # 假设市场平均振幅3%
            market_avg_volume_ratio = 1.0  # 假设市场平均量比为1
        
        # 计算相对强度
        # ✅ 保留：相对收益（最佳新特征，相关性0.106）
        features['relative_gain_vs_market'] = stock_gain - market_gain
        
        # ✅ 保留：相对换手率（相关性0.062，有一定预测力）
        features['relative_turnover_vs_market'] = stock_avg_turnover / (market_avg_turnover + 1e-9)
        
        # ❌ 移除：相对振幅 - 已移除（V2.4.1）
        # 与`prior_total_amplitude`完全相关（都是0.097），保留前者即可
        # features['relative_amplitude_vs_market'] = stock_avg_amplitude / (market_avg_amplitude + 1e-9)
        
        # ❌ 移除：相对量比 - 已移除（V2.4.1）
        # 诊断发现：相关性-0.035，与收益负相关，可能是噪音
        # features['relative_volume_ratio_vs_market'] = stock_avg_volume_ratio / (market_avg_volume_ratio + 1e-9)
        
        return features
    
    @staticmethod
    def extract_all_features(
        instrument: str,
        T0_date: pd.Timestamp,
        T3_date: pd.Timestamp,
        kline_df: pd.DataFrame,
        stock_char_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        提取所有特征
        
        Args:
            instrument: 股票代码
            T0_date: T0涨停日期
            T3_date: T+3日期（预测日）
            kline_df: K线数据（包含MA）
            stock_char_df: 股性特征数据
            sentiment_df: 市场情绪数据
        
        Returns:
            包含所有特征的字典，如果数据不足返回None
        """
        # 获取该股票的K线数据
        stock_kline = kline_df[kline_df['instrument'] == instrument].copy()
        stock_kline = stock_kline.sort_values('date').reset_index(drop=True)
        
        # 找到T0在数据中的位置
        T0_mask = stock_kline['date'] == T0_date
        if not T0_mask.any():
            return None
        
        T0_idx = stock_kline[T0_mask].index[0]
        
        # 获取T0~T+3的数据
        if T0_idx + 4 > len(stock_kline):
            return None
        
        kline_4days = stock_kline.iloc[T0_idx:T0_idx+4]
        T0_close = kline_4days.iloc[0]['close']
        
        # 计算该股票的20日平均成交量
        if T0_idx < 20:
            stock_avg_volume = stock_kline['volume'].iloc[:T0_idx].mean()
        else:
            stock_avg_volume = stock_kline['volume'].iloc[T0_idx-20:T0_idx].mean()
        
        # 提取各类特征
        all_features = {}
        
        # 1. 形态特征
        pattern_features = ConsolidationFeatureExtractor.extract_pattern_features(
            kline_4days, T0_close
        )
        all_features.update(pattern_features)
        
        # 2. 均线特征
        ma_features = ConsolidationFeatureExtractor.extract_ma_features(
            stock_kline, T0_idx
        )
        all_features.update(ma_features)
        
        # 3. 量能特征
        volume_features = ConsolidationFeatureExtractor.extract_volume_features(
            kline_4days, stock_avg_volume
        )
        all_features.update(volume_features)
        
        # 4. 股性特征
        stock_char_features = ConsolidationFeatureExtractor.extract_stock_character_features(
            instrument, T3_date, stock_char_df
        )
        all_features.update(stock_char_features)
        
        # 5. 市场特征（已被移除，保留代码结构以便将来需要）
        market_features = ConsolidationFeatureExtractor.extract_market_features(
            T0_date, T3_date, sentiment_df
        )
        all_features.update(market_features)
        
        # 6. 前期表现特征（新增 - V2.4）
        prior_performance_features = ConsolidationFeatureExtractor.extract_prior_performance_features(
            kline_4days, T0_close
        )
        all_features.update(prior_performance_features)
        
        # 7. 相对强度特征（新增 - V2.4）
        relative_strength_features = ConsolidationFeatureExtractor.extract_relative_strength_features(
            stock_kline, T0_idx, index_kline_df=None
        )
        all_features.update(relative_strength_features)
        
        return all_features


def test_feature_extractor():
    """测试特征提取器"""
    import sys
    from pathlib import Path
    
    # 添加项目根目录到路径
    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT))
    
    from consolidation_data_loader import ConsolidationDataLoader
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("\n" + "=" * 80)
    logging.info("测试特征提取器")
    logging.info("=" * 80)
    
    # 加载数据
    loader = ConsolidationDataLoader()
    limit_up_df, kline_df, stock_char_df, sentiment_df = loader.load_all_data()
    
    # 测试前3个样本
    logging.info("\n测试前3个样本的特征提取...")
    test_samples = limit_up_df.head(3)
    
    for idx, row in test_samples.iterrows():
        instrument = row['instrument']
        T0_date = row['date']
        
        # 计算T+3日期
        T3_date = loader.get_future_trading_date(T0_date, 3)
        if T3_date is None:
            logging.info(f"\n{instrument} {T0_date.date()} - T+3日期不存在")
            continue
        
        # 提取特征
        features = ConsolidationFeatureExtractor.extract_all_features(
            instrument, T0_date, T3_date,
            kline_df, stock_char_df, sentiment_df
        )
        
        if features is None:
            logging.info(f"\n{instrument} {T0_date.date()} - 数据不足")
            continue
        
        logging.info(f"\n{instrument} {T0_date.date()} 的特征：")
        logging.info(f"  总特征数：{len(features)}")
        
        # 按类别统计特征
        pattern_count = sum(1 for k in features if k in [
            'low_slope', 'close_center_vs_T0', 'avg_close_position', 'avg_body_ratio',
            'bullish_count', 'amplitude_avg', 'amplitude_max', 'amplitude_min',
            'amplitude_std', 'upper_shadow_avg', 'lower_shadow_avg', 'high_slope',
            'close_slope', 'range_convergence', 'T1_gain', 'T2_gain', 'T3_gain',
            'cumulative_gain', 'max_drawdown', 'price_volatility', 'pattern_score'
        ])
        
        ma_count = sum(1 for k in features if k in [
            'ma5_bias', 'ma10_bias', 'ma20_bias', 'multi_bull_degree',
            'ma5_ma10_divergence', 'ma5_ma20_divergence', 'ma10_ma20_divergence',
            'days_above_ma5', 'days_above_ma10', 'days_above_ma20',
            'ma5_slope', 'avg_ma5_position'
        ])
        
        volume_count = sum(1 for k in features if k in [
            'avg_volume_ratio', 'volume_decay_rate', 'total_turnover', 'volume_vs_T0',
            'price_volume_corr', 'volume_volatility', 'max_volume_ratio', 'min_volume_ratio'
        ])
        
        stock_char_count = sum(1 for k in features if k in [
            'volatility_60d', 'up_day_ratio_60d', 'rebound_speed', 'amplitude_avg_60d',
            'up_body_sum_ratio_60d', 'volume_price_correlation', 'turnover_rate'
        ])
        
        # ❌ 市场情绪特征已被移除
        # market_count = sum(1 for k in features if k in [
        #     'market_activity_avg', 'limit_up_count_avg', 'sentiment_score_avg',
        #     'median_gain_avg', 'market_trend'
        # ])
        
        logging.info(f"  形态特征：{pattern_count} 个")
        logging.info(f"  均线特征：{ma_count} 个")
        logging.info(f"  量能特征：{volume_count} 个")
        logging.info(f"  股性特征：{stock_char_count} 个")
        # logging.info(f"  市场特征：{market_count} 个")  # ❌ 已移除
        
        # 显示一些关键特征
        logging.info(f"  关键特征示例：")
        logging.info(f"    pattern_score: {features.get('pattern_score', 0):.3f}")
        logging.info(f"    cumulative_gain: {features.get('cumulative_gain', 0):.3f}")
        logging.info(f"    ma5_bias: {features.get('ma5_bias', 0):.3f}")
        logging.info(f"    up_day_ratio_60d: {features.get('up_day_ratio_60d', 0):.3f}")
        # logging.info(f"    sentiment_score_avg: {features.get('sentiment_score_avg', 0):.1f}")  # ❌ 已移除


if __name__ == '__main__':
    test_feature_extractor()
