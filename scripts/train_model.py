import os
import sys
import pandas as pd
import numpy as np
import logging
import joblib
import sqlite3
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score
import lightgbm as lgb
import xgboost as xgb
import json
from tqdm import tqdm
from typing import Optional

# 添加脚本目录到路径，以便导入 enhanced_features_service
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# 配置日志
base_dir = os.path.dirname(script_dir)
logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, 'train_model.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class SecondWaveModelTrainer:
    def __init__(self, data_dir='data', model_dir='models'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # 设置基础目录
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.enhanced_features_dir = os.path.join(self.base_dir, 'data', 'raw')
        
    def prepare_local_features(self):
        """
        使用本地已处理的 K 线数据进行特征工程
        代替 BigQuant 在线因子，以解决配额不足问题
        """
        logging.info("正在从本地数据构建特征...")
        
        # 1. 加载本地标签数据
        # 【修复】数据库路径从 data/processed 改为 data
        db_path = os.path.join(self.base_dir, 'data', 'historical_patterns.db')
        if not os.path.exists(db_path):
            logging.error(f"未找到模式数据库: {db_path}")
            logging.error("请先运行 analyze_patterns.py 生成历史模式数据")
            return None
            
        conn = sqlite3.connect(db_path)
        labels_df = pd.read_sql("SELECT * FROM stock_patterns", conn)
        conn.close()
        
        if labels_df.empty:
            logging.error("标签数据为空")
            logging.error("请先运行 analyze_patterns.py 分析历史涨停数据以生成训练标签")
            return None
            
        logging.info(f"本地标签数据加载完成: {len(labels_df)} 条记录")
        labels_df['trigger_date'] = pd.to_datetime(labels_df['trigger_date'], format='ISO8601')
        
        # 2. 加载用于训练的“日线+派生特征”数据
        # 优先使用 data/raw/features/ 下最新生成的 features_kline_stock_character_*.csv（包含15个股性/量价特征）
        # 若不存在，则回退到 data/raw/kline/ 下最新的 kline_*.csv / kline_all.csv
        def _get_latest_file(directory: str, prefix: str, suffix: str = ".csv") -> Optional[str]:
            if not os.path.exists(directory):
                return None
            files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(suffix)]
            if not files:
                return None
            latest = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
            return os.path.join(directory, latest)

        # 2.1 优先使用特征文件
        features_dir = os.path.join(self.base_dir, 'data', 'raw', 'features')
        features_path = _get_latest_file(features_dir, 'features_kline_stock_character_')
        if features_path:
            kline_path = features_path
            logging.info(f"使用最新股性/量价特征文件: {kline_path}")
        else:
            # 2.2 回退：加载处理后的 K 线数据以提取更多特征
            # 优先使用 data/processed/kline_processed.csv，如果不存在则从 data/raw/kline/ 查找最新文件
            kline_path = f"{self.data_dir}/kline_processed.csv"
            if not os.path.exists(kline_path):
                # 尝试从 data/raw/kline/ 目录查找最新文件
                kline_dir = os.path.join(self.base_dir, 'data', 'raw', 'kline')
                if os.path.exists(kline_dir):
                    kline_files = [f for f in os.listdir(kline_dir) if f.startswith('kline_') and f.endswith('.csv')]
                    if kline_files:
                        # 按文件修改时间排序，取最新的
                        latest_file = max(kline_files, key=lambda f: os.path.getmtime(os.path.join(kline_dir, f)))
                        kline_path = os.path.join(kline_dir, latest_file)
                        logging.info(f"使用最新K线文件: {kline_path}")
                    else:
                        logging.error("未找到K线数据文件")
                        return None
                else:
                    logging.error("未找到K线数据目录")
                    return None
            
        kline_df = pd.read_csv(kline_path)
        kline_df['date'] = pd.to_datetime(kline_df['date'], format='ISO8601')
        
        # 3. 加载技术分析因子（优先使用 BigQuant 的专业计算数据）
        ta_factors_path = os.path.join(self.base_dir, 'data', 'raw', 'ta_factors.csv')
        ta_factors_df = None
        
        if os.path.exists(ta_factors_path):
            logging.info("加载 BigQuant 技术分析因子数据...")
            try:
                ta_factors_df = pd.read_csv(ta_factors_path)
                ta_factors_df['date'] = pd.to_datetime(ta_factors_df['date'], format='ISO8601')
                logging.info(f"✓ 技术分析因子加载完成: {len(ta_factors_df)} 条记录")
                
                # 合并技术指标到K线数据
                # 【修正】BigQuant cn_stock_factors_ta 实际有113个字段！
                ta_factors_df = ta_factors_df.rename(columns={
                    # 简单移动平均线 (SMA)
                    'sma_5': 'ma5',
                    'sma_10': 'ma10',
                    'sma_20': 'ma20',
                    'sma_60': 'ma60',
                    'sma_120': 'ma120',
                    'sma_250': 'ma250',
                    # MACD（标准12-26-9参数）
                    'macd_diff_12_26_9': 'macd_dif',
                    'macd_dea_12_26_9': 'macd_dea',
                    'macd_hist_12_26_9': 'macd_hist',
                    # KDJ（标准9-3-3参数）
                    'kdj_k_9_3_3': 'kdj_k',
                    'kdj_d_9_3_3': 'kdj_d',
                    'kdj_j_9_3_3': 'kdj_j',
                    # 布林带（标准20日，2倍标准差）
                    'bbands_upper_20_2': 'boll_up',
                    'bbands_middle_20_2': 'boll_mid',
                    'bbands_lower_20_2': 'boll_down'
                })
                
                # 选择技术指标字段（充分利用113个字段中的精华）
                ta_cols = [
                    'instrument', 'date',
                    # 基础移动平均线（SMA）
                    'ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250',
                    # 指数移动平均线（EMA）
                    'ema_5', 'ema_10', 'ema_20', 'ema_60', 'ema_120', 'ema_250',
                    # 加权移动平均线（WMA）
                    'wma_5', 'wma_10', 'wma_20', 'wma_60', 'wma_120', 'wma_250',
                    # MACD 系列（多参数）
                    'macd_dif', 'macd_dea', 'macd_hist',  # 标准12-26-9
                    'macd_diff_5_20_5', 'macd_dea_5_20_5', 'macd_hist_5_20_5',  # 快速参数
                    # 乖离率（BIAS）
                    'bias_5', 'bias_10', 'bias_20', 'bias_60', 'bias_120', 'bias_250',
                    # KDJ（多参数）
                    'kdj_k', 'kdj_d', 'kdj_j',  # 标准9-3-3
                    'kdj_k_5_3_3', 'kdj_d_5_3_3', 'kdj_j_5_3_3',  # 快速参数
                    # RSI（多周期）
                    'rsi_6', 'rsi_12', 'rsi_24', 'rsi_48', 'rsi_60', 'rsi_120',
                    # 布林带
                    'boll_up', 'boll_mid', 'boll_down',  # 标准20日
                    'bbands_upper_10_2', 'bbands_middle_10_2', 'bbands_lower_10_2',  # 10日
                    # CCI 指标
                    'cci_14', 'cci_20', 'cci_60',
                    # ATR 真实波动幅度
                    'atr_14', 'atr_20', 'atr_60',
                    # OBV 能量潮
                    'obv', 'maobv_20',
                    # TRIX 三重指数平滑
                    'trix_12_9', 'trix_20_12'
                ]
                ta_subset = ta_factors_df[[col for col in ta_cols if col in ta_factors_df.columns]].copy()
                
                available_count = len([c for c in ta_cols if c in ta_factors_df.columns])
                logging.info(f"✓ 可用的技术指标字段: {available_count}/{len(ta_cols)-2} 个")
                
                # 左连接：保留所有K线数据，匹配技术指标
                kline_df = pd.merge(
                    kline_df,
                    ta_subset,
                    on=['instrument', 'date'],
                    how='left',
                    suffixes=('', '_ta')
                )
                
                # 如果K线数据本身有这些字段，优先使用技术分析因子的值
                for col in ['ma5', 'ma10', 'ma20', 'ma60', 'rsi']:
                    if f'{col}_ta' in kline_df.columns:
                        kline_df[col] = kline_df[f'{col}_ta'].fillna(kline_df.get(col, 0))
                        kline_df.drop(columns=[f'{col}_ta'], inplace=True)
                
                logging.info("✓ 技术分析因子已合并到K线数据")
            except Exception as e:
                logging.warning(f"加载技术分析因子失败: {e}，将使用手动计算")
                ta_factors_df = None
        else:
            logging.warning(f"未找到技术分析因子文件: {ta_factors_path}")
            logging.warning("建议运行: python scripts/download_ta_factors.py")
        
        # 4. 如果技术指标仍然缺失，手动计算
        required_fields = ['rsi', 'pct_change', 'ma5', 'ma10', 'ma20', 'ma60']
        missing_fields = [f for f in required_fields if f not in kline_df.columns]
        
        if missing_fields:
            logging.info(f"K线数据缺少技术指标，开始手动计算: {missing_fields}")
            
            # 确保有前复权价格字段
            if 'close_qfq' not in kline_df.columns:
                if 'adjust_factor' in kline_df.columns:
                    kline_df['close_qfq'] = kline_df['close'] * kline_df['adjust_factor']
                else:
                    kline_df['close_qfq'] = kline_df['close']
            
            # 按股票和日期排序
            kline_df = kline_df.sort_values(['instrument', 'date']).reset_index(drop=True)
            
            # 按股票分组计算技术指标
            def calc_indicators(group):
                # 计算涨跌幅
                if 'pct_change' not in group.columns:
                    group['pct_change'] = group['close'].pct_change(fill_method=None) * 100
                
                # 计算移动平均线
                if 'ma5' not in group.columns:
                    group['ma5'] = group['close_qfq'].rolling(window=5, min_periods=1).mean()
                if 'ma10' not in group.columns:
                    group['ma10'] = group['close_qfq'].rolling(window=10, min_periods=1).mean()
                if 'ma20' not in group.columns:
                    group['ma20'] = group['close_qfq'].rolling(window=20, min_periods=1).mean()
                if 'ma60' not in group.columns:
                    group['ma60'] = group['close_qfq'].rolling(window=60, min_periods=1).mean()
                
                # 计算RSI
                if 'rsi' not in group.columns:
                    delta = group['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                    rs = gain / (loss + 1e-10)
                    group['rsi'] = 100 - (100 / (1 + rs))
                
                return group
            
            # 使用 tqdm 显示进度
            logging.info(f"开始计算技术指标，共 {kline_df['instrument'].nunique()} 只股票...")
            tqdm.pandas(desc="计算技术指标")
            kline_df = kline_df.groupby('instrument', group_keys=False).progress_apply(calc_indicators)
            logging.info("✓ 技术指标计算完成")
        else:
            # 确保有前复权价格字段
            if 'close_qfq' not in kline_df.columns:
                if 'adjust_factor' in kline_df.columns:
                    kline_df['close_qfq'] = kline_df['close'] * kline_df['adjust_factor']
                else:
                    kline_df['close_qfq'] = kline_df['close']
        
        # 5. 提取触发当日特征
        # 【扩展】合并更多技术指标 + 股性/量价派生特征（15个）
        base_features = ['instrument', 'date', 'close_qfq', 'pct_change']

        stock_character_features = [
            # 股性（7）
            'volatility_60d', 'limit_up_frequency', 'second_wave_history',
            'amplitude_avg_60d', 'hot_stock_days', 'concept_rotation_count', 'rebound_speed',
            # 量价（8）
            'volume_price_correlation', 'volume_increase_ratio', 'volume_price_divergence',
            'shrink_limit_up', 'turnover_rate', 'volume_continuity', 'volume_pattern', 'volume_ma_ratio',
            # 股性偏好增强（A）+ 量价偏好增强（B）
            'up_day_ratio_60d', 'up_body_sum_ratio_60d_log', 'net_body_strength_60d',
            'up_volume_ratio_60d', 'up_amount_ratio_60d', 'turnover_trend_20d'
        ]
        
        # 动态添加可用的技术指标字段
        optional_features = [
            'ma5', 'ma10', 'ma20', 'ma60', 'ma120',
            'rsi', 'rsi_6', 'rsi_12', 'rsi_24',
            'macd', 'macd_dea', 'macd_dif',
            'boll_up', 'boll_mid', 'boll_down',
            'kdj_k', 'kdj_d', 'kdj_j',
            'cci', 'cci_20',
            'willr', 'willr_14'
        ]
        
        features_to_add = (
            base_features
            + [f for f in stock_character_features if f in kline_df.columns]
            + [f for f in optional_features if f in kline_df.columns]
        )
        kline_subset = kline_df[features_to_add].copy()
        kline_subset['date'] = pd.to_datetime(kline_subset['date'], format='ISO8601')
        
        # 计算移动平均线偏离度（如果有均线数据）
        for ma_period in [5, 10, 20, 60, 120]:
            ma_col = f'ma{ma_period}'
            if ma_col in kline_subset.columns:
                kline_subset[f'bias_ma{ma_period}'] = (kline_subset['close_qfq'] / kline_subset[ma_col]) - 1
        
        final_df = pd.merge(
            labels_df,
            kline_subset,
            left_on=['symbol', 'trigger_date'],
            right_on=['instrument', 'date'],
            how='inner'
        )
        
        # 6. 合并筹码数据 (如果存在)
        chips_path = os.path.join(self.base_dir, 'data', 'raw', 'chips_all.csv')
        if os.path.exists(chips_path):
            logging.info("加载筹码分布数据...")
            chips_df = pd.read_csv(chips_path)
            chips_df['date'] = pd.to_datetime(chips_df['date'], format='ISO8601')
            
            final_df = pd.merge(
                final_df,
                chips_df[['instrument', 'date', 'avg_cost', 'win_percent', 'concentration']],
                left_on=['symbol', 'trigger_date'],
                right_on=['instrument', 'date'],
                how='left'
            )
            
            # 计算衍生筹码特征
            # 价格/成本比：反映获利空间
            final_df['price_to_cost'] = (final_df['close_qfq'] / final_df['avg_cost']) - 1
            logging.info("已集成筹码特征")
        else:
            logging.warning("未找到筹码数据，将跳过筹码特征")
            final_df['win_percent'] = 0
            final_df['concentration'] = 0
            final_df['price_to_cost'] = 0
        
        # 7. 合并龙虎榜数据 (如果存在)
        dragon_path = os.path.join(self.base_dir, 'data', 'raw', 'limit_up', 'dragon_list.csv')
        if os.path.exists(dragon_path):
            logging.info("加载龙虎榜数据...")
            dragon_df = pd.read_csv(dragon_path)
            dragon_df['date'] = pd.to_datetime(dragon_df['date'], format='ISO8601')
            
            # 汇总当日净买入额 (同一天可能有多条记录)
            dragon_summary = dragon_df.groupby(['instrument', 'date'])['net_buy_amount'].sum().reset_index()
            
            final_df = pd.merge(
                final_df,
                dragon_summary,
                left_on=['symbol', 'trigger_date'],
                right_on=['instrument', 'date'],
                how='left'
            )
            final_df['net_buy_amount'] = final_df['net_buy_amount'].fillna(0)
            logging.info("已集成龙虎榜特征")
        else:
            logging.warning("未找到龙虎榜数据，将跳过龙虎榜特征")
            final_df['net_buy_amount'] = 0
        
        # 8. 【新增】加载市场情绪数据
        sentiment_db_path = os.path.join(self.base_dir, 'data', 'market_sentiment.db')
        if os.path.exists(sentiment_db_path):
            logging.info("加载市场情绪数据...")
            try:
                conn = sqlite3.connect(sentiment_db_path)
                sentiment_df = pd.read_sql("SELECT * FROM market_sentiment", conn)
                conn.close()
                
                sentiment_df['crawl_date'] = pd.to_datetime(sentiment_df['crawl_date'])
                
                final_df = pd.merge(
                    final_df,
                    sentiment_df[['crawl_date', 'sentiment_score', 'limit_up_count', 'limit_up_real_count']],
                    left_on='trigger_date',
                    right_on='crawl_date',
                    how='left'
                )
                final_df['sentiment_score'] = final_df['sentiment_score'].fillna(0)
                final_df['limit_up_count'] = final_df['limit_up_count'].fillna(0)
                final_df['limit_up_real_count'] = final_df['limit_up_real_count'].fillna(0)
                logging.info("已集成市场情绪特征")
            except Exception as e:
                logging.warning(f"加载市场情绪数据失败: {e}")
                final_df['sentiment_score'] = 0
                final_df['limit_up_count'] = 0
                final_df['limit_up_real_count'] = 0
        else:
            logging.warning("未找到市场情绪数据，将跳过市场情绪特征")
            final_df['sentiment_score'] = 0
            final_df['limit_up_count'] = 0
            final_df['limit_up_real_count'] = 0
        
        # 9. 【新增】加载增强特征（热度、概念、竞价）- 向量化批量操作
        try:
            logging.info("加载增强特征数据...")
            from enhanced_features_service import EnhancedFeaturesService
            enhanced_service = EnhancedFeaturesService(self.enhanced_features_dir)
            
            # 批量提取增强特征（向量化操作，速度快10-50倍）
            total_samples = len(final_df)
            logging.info(f"开始批量提取增强特征，共 {total_samples} 个样本...")
            
            # 打印调试信息
            logging.info(f"final_df 列: {final_df.columns.tolist()[:10]}...")  # 只打印前10列
            logging.info(f"final_df 示例行: symbol={final_df['symbol'].iloc[0]}, trigger_date={final_df['trigger_date'].iloc[0]}")
            
            final_df = enhanced_service.get_batch_features(final_df)

            # === 训练/预测字段名对齐（避免静默丢特征）===
            # 历史版本可能使用 hotness_rank，当前增强特征服务使用 hot_rank
            if 'hotness_rank' in final_df.columns and 'hot_rank' not in final_df.columns:
                final_df = final_df.rename(columns={'hotness_rank': 'hot_rank'})

            # 关键增强特征的“默认值占比”告警：防止因为 symbol/date 对不上导致整列全默认
            def _warn_default_ratio(col: str, default_value, ratio_threshold: float = 0.98):
                if col not in final_df.columns:
                    logging.warning(f"⚠️ 关键特征列缺失: {col}（可能导致训练静默丢特征）")
                    return
                s = final_df[col]
                try:
                    total = len(s)
                    if total == 0:
                        return
                    default_ratio = float((s == default_value).sum()) / float(total)
                    if default_ratio >= ratio_threshold:
                        logging.warning(
                            f"⚠️ 特征 {col} 默认值占比过高: {default_ratio:.1%} (default={default_value})，"
                            f"请检查增强特征 merge 的 symbol/date 是否对齐"
                        )
                except Exception:
                    # 告警本身不应影响训练流程
                    return

            _warn_default_ratio('hot_rank', 999)
            _warn_default_ratio('concept_count', 0)
            _warn_default_ratio('auction_strength', 0)
            
            logging.info(f"✓ 增强特征批量提取完成，共处理 {total_samples} 个样本")
        except Exception as e:
            import traceback
            logging.warning(f"加载增强特征失败: {e}")
            logging.warning(f"详细错误: {traceback.format_exc()}")
            # 添加默认值
            for col in ['hot_rank', 'hot_duration', 'concept_count', 'concept_avg_gain', 
                       'concept_max_gain', 'concept_momentum', 'concept_resonance',
                       'auction_volume_ratio', 'auction_price_gap', 'auction_turnover', 'auction_strength']:
                final_df[col] = 0
            
        logging.info(f"本地特征构建完成，样本数: {len(final_df)}, 特征数: {len(final_df.columns)}")
        return final_df

    def train(self, df):
        """训练 LightGBM 模型"""
        if df is None or df.empty:
            logging.error("训练数据为空，无法训练")
            return
            
        # 准备特征
        # 【完整版】充分利用 BigQuant 的113个技术指标字段！
        feature_cols = [
            # 基础特征（4个）
            'consecutive_count', 'volume_ratio', 'pct_change', 'max_drawdown',
            
            # === 技术指标特征（来自 BigQuant cn_stock_factors_ta）===
            
            # 移动平均线偏离度（手工计算，5个）
            'bias_ma5', 'bias_ma10', 'bias_ma20', 'bias_ma60', 'bias_ma120',
            
            # BigQuant 乖离率（BIAS，6个）
            'bias_5', 'bias_10', 'bias_20', 'bias_60', 'bias_120', 'bias_250',
            
            # EMA 指数移动平均线（6个）
            'ema_5', 'ema_10', 'ema_20', 'ema_60', 'ema_120', 'ema_250',
            
            # WMA 加权移动平均线（6个）
            'wma_5', 'wma_10', 'wma_20', 'wma_60', 'wma_120', 'wma_250',
            
            # MACD 标准参数（3个）
            'macd_dif', 'macd_dea', 'macd_hist',
            
            # MACD 快速参数（3个）
            'macd_diff_5_20_5', 'macd_dea_5_20_5', 'macd_hist_5_20_5',
            
            # KDJ 标准参数（3个）
            'kdj_k', 'kdj_d', 'kdj_j',
            
            # KDJ 快速参数（3个）
            'kdj_k_5_3_3', 'kdj_d_5_3_3', 'kdj_j_5_3_3',
            
            # RSI 多周期（6个）
            'rsi_6', 'rsi_12', 'rsi_24', 'rsi_48', 'rsi_60', 'rsi_120',
            
            # 布林带 20日（3个）
            'boll_up', 'boll_mid', 'boll_down',
            
            # 布林带 10日（3个）
            'bbands_upper_10_2', 'bbands_middle_10_2', 'bbands_lower_10_2',
            
            # CCI 指标（3个）
            'cci_14', 'cci_20', 'cci_60',
            
            # ATR 真实波动幅度（3个）
            'atr_14', 'atr_20', 'atr_60',
            
            # OBV 能量潮（2个）
            'obv', 'maobv_20',
            
            # TRIX 三重指数平滑（2个）
            'trix_12_9', 'trix_20_12',
            
            # === 其他特征 ===
            
            # 筹码特征（3个）
            'win_percent', 'concentration', 'price_to_cost', 
            
            # 龙虎榜特征（1个）
            'net_buy_amount',
            
            # 市场情绪特征（3个）
            'sentiment_score', 'limit_up_count', 'limit_up_real_count',
            
            # 增强特征 - 热度（2个）
            'hot_rank', 'hot_duration',
            
            # 增强特征 - 概念（5个）
            'concept_count', 'concept_avg_gain', 'concept_max_gain', 
            'concept_momentum', 'concept_resonance',
            
            # 增强特征 - 竞价（4个）
            'auction_volume_ratio', 'auction_price_gap', 
            'auction_turnover', 'auction_strength'
            ,
            # === 新增：股性/量价特征（15个）===
            'volatility_60d', 'limit_up_frequency', 'second_wave_history',
            'amplitude_avg_60d', 'hot_stock_days', 'concept_rotation_count', 'rebound_speed',
            'volume_price_correlation', 'volume_increase_ratio', 'volume_price_divergence',
            'shrink_limit_up', 'turnover_rate', 'volume_continuity', 'volume_pattern', 'volume_ma_ratio',
            # === 新增：股性偏好增强（A）+ 量价偏好增强（B）（6个）===
            'up_day_ratio_60d', 'up_body_sum_ratio_60d_log', 'net_body_strength_60d',
            'up_volume_ratio_60d', 'up_amount_ratio_60d', 'turnover_trend_20d'
        ]
        
        # 特征总数统计（预期）：
        # 基础特征: 4
        # 技术指标: 5+6+6+6+3+3+3+3+6+3+3+3+3+2+2 = 61
        # 筹码: 3
        # 龙虎榜: 1
        # 市场情绪: 3
        # 增强特征: 2+5+4 = 11
        # 股性/量价（新增）: 15
        # 股性偏好增强（A）+ 量价偏好增强（B）（新增）: 6
        # 总计: 83 + 15 + 6 = 104 个特征（如果数据列齐全）
        
        # 确保所有列都存在
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[feature_cols]
        y = df['second_wave_confirmed']
        
        # 处理无穷值和缺失值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. 训练 LightGBM
        logging.info(f"开始训练 LightGBM... 特征数: {len(feature_cols)}")
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5
        }
        
        train_data_lgb = lgb.Dataset(X_train, label=y_train)
        test_data_lgb = lgb.Dataset(X_test, label=y_test, reference=train_data_lgb)
        
        lgb_model = lgb.train(
            lgb_params,
            train_data_lgb,
            num_boost_round=1000,
            valid_sets=[train_data_lgb, test_data_lgb],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        # 2. 训练 XGBoost
        logging.info(f"开始训练 XGBoost...")
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # 评估集成效果 (简单平均)
        lgb_pred = lgb_model.predict(X_test)
        xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
        ensemble_pred = (lgb_pred + xgb_pred) / 2
        
        auc_lgb = roc_auc_score(y_test, lgb_pred)
        auc_xgb = roc_auc_score(y_test, xgb_pred)
        auc_ensemble = roc_auc_score(y_test, ensemble_pred)
        
        logging.info(f"LightGBM AUC: {auc_lgb:.4f}")
        logging.info(f"XGBoost AUC: {auc_xgb:.4f}")
        logging.info(f"Ensemble AUC: {auc_ensemble:.4f}")
        
        # 保存模型
        lgb_path = f"{self.model_dir}/second_wave_lgb.model"
        xgb_path = f"{self.model_dir}/second_wave_xgb.model"
        joblib.dump(lgb_model, lgb_path)
        joblib.dump(xgb_model, xgb_path)
        
        # 保存特征列表
        feature_names_path = f"{self.model_dir}/feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(feature_cols, f)
            
        logging.info(f"模型集成已保存至: {self.model_dir}")
        
if __name__ == "__main__":
    trainer = SecondWaveModelTrainer()
    data = trainer.prepare_local_features()
    if data is not None:
        trainer.train(data)
