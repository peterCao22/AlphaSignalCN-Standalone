"""
股性与量价配合特征工程脚本

功能：
1. 计算7个股性特征（波动率/涨停频率/历史二波成功率等）
2. 计算8个量价配合特征（量价相关性/量价背离/缩量涨停等）
3. 保存特征数据到CSV文件

数据来源：
- K线数据（已有）
- 热度排名（已有）
- 概念数据（已有）
- 历史二波记录（已有）

使用方法：
python scripts/enhanced_stock_character_features.py
"""

import sys
from pathlib import Path

# 统一以本仓库根目录为准（AlphaSignalCN-Standalone）
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StockCharacterFeatureEngineer:
    """股性与量价特征工程"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.kline_path = self.data_dir / 'raw' / 'kline' / 'kline_all.csv'
        self.hot_rank_path = self.data_dir / 'raw' / 'hot_rank.csv'
        self.concept_path = self.data_dir / 'raw' / 'concept_component.csv'
        self.db_path = self.data_dir / 'historical_patterns.db'
        self.second_wave_history_map = {}
        # 输出到独立目录 + 时间戳文件名，避免被补K线逻辑误选中/覆盖
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_path = (
            self.data_dir
            / 'raw'
            / 'features'
            / f'features_kline_stock_character_{timestamp}.csv'
        )
        
    def load_data(self):
        """加载所有数据"""
        logging.info("正在加载数据...")
        
        # 加载K线数据
        logging.info(f"  读取K线数据: {self.kline_path}")
        self.kline_df = pd.read_csv(self.kline_path)
        self.kline_df['date'] = pd.to_datetime(self.kline_df['date'], format='ISO8601')
        logging.info(f"  ✓ K线数据: {len(self.kline_df)} 条记录, {len(self.kline_df['instrument'].unique())} 只股票")
        
        # 加载热度排名（如果存在）
        if self.hot_rank_path.exists():
            logging.info(f"  读取热度排名: {self.hot_rank_path}")
            self.hot_rank_df = pd.read_csv(self.hot_rank_path)
            self.hot_rank_df['date'] = pd.to_datetime(self.hot_rank_df['date'], format='ISO8601')
            logging.info(f"  ✓ 热度排名: {len(self.hot_rank_df)} 条记录")
        else:
            logging.warning(f"  ⚠ 热度排名文件不存在: {self.hot_rank_path}")
            self.hot_rank_df = None
            
        # 加载概念数据（如果存在）
        if self.concept_path.exists():
            logging.info(f"  读取概念数据: {self.concept_path}")
            self.concept_df = pd.read_csv(self.concept_path)
            self.concept_df['date'] = pd.to_datetime(self.concept_df['date'], format='ISO8601')
            logging.info(f"  ✓ 概念数据: {len(self.concept_df)} 条记录")
        else:
            logging.warning(f"  ⚠ 概念数据文件不存在: {self.concept_path}")
            self.concept_df = None

        # 预加载二波历史成功率（一次性统计，避免每只股票都查库）
        self._load_second_wave_history_map()
            
        logging.info("✓ 数据加载完成\n")

    def _load_second_wave_history_map(self):
        """
        从 historical_patterns.db 一次性统计每只股票的历史二波成功率（second_wave_history）

        说明：
        - 训练脚本 `train_model.py` 使用的标签表为 `stock_patterns`（且包含 second_wave_confirmed）
        - 这里直接复用 `stock_patterns` 来做“市场记忆/股性”统计，避免之前查询不存在的 historical_patterns 表导致全为0.5
        """
        self.second_wave_history_map = {}
        if not self.db_path.exists():
            logging.warning(f"  ⚠ 历史模式数据库不存在: {self.db_path}，second_wave_history 将使用默认0.5")
            return

        try:
            conn = sqlite3.connect(self.db_path)
            # 确认表存在
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'",
                conn
            )["name"].tolist()

            if "stock_patterns" not in tables:
                logging.warning(
                    f"  ⚠ 数据库中不存在 stock_patterns 表（tables={tables}），second_wave_history 将使用默认0.5"
                )
                conn.close()
                return

            df = pd.read_sql_query(
                "SELECT symbol, second_wave_confirmed FROM stock_patterns",
                conn
            )
            conn.close()

            if df.empty:
                logging.warning("  ⚠ stock_patterns 为空，second_wave_history 将使用默认0.5")
                return

            # 统计每只股票：样本数、成功数、成功率
            agg = df.groupby("symbol")["second_wave_confirmed"].agg(["count", "sum"]).reset_index()
            # 少样本（<3）不可信，统一回退0.5
            agg["rate"] = np.where(agg["count"] >= 3, agg["sum"] / agg["count"], 0.5)
            self.second_wave_history_map = dict(zip(agg["symbol"], agg["rate"]))

            logging.info(
                f"  ✓ second_wave_history 统计完成: {len(self.second_wave_history_map)} 只股票（来源: stock_patterns）"
            )
        except Exception as e:
            logging.warning(f"  ⚠ second_wave_history 统计失败: {e}，将使用默认0.5")
            self.second_wave_history_map = {}
        
    def calculate_limit_up(self, group):
        """计算是否涨停"""
        stock_code = group['instrument'].iloc[0].split('.')[0]
        
        if stock_code.startswith(('92', '43', '8')):
            limit = 29.5  # 北交所30%
        elif stock_code.startswith(('300', '301', '688', '689')):
            limit = 19.5  # 创业板/科创板20%
        else:
            limit = 9.5   # 主板10%
            
        # 计算pct_change（如果不存在）
        if 'pct_change' not in group.columns:
            group['pct_change'] = (group['close'] - group['close'].shift(1)) / group['close'].shift(1) * 100
            
        group['is_limit_up'] = (group['pct_change'] >= limit).astype(int)
        return group
        
    def get_historical_second_wave_rate(self, symbol):
        """从历史数据库查询个股的二波成功率"""
        # 使用预加载的缓存，避免每只股票都查库
        return float(self.second_wave_history_map.get(symbol, 0.5))
            
    def calculate_rebound_speed(self, group):
        """计算回调反弹速度"""
        is_limit_up = group['is_limit_up'].values
        speeds = []
        
        for i in range(1, len(group)):
            if is_limit_up[i] == 1:  # 当天涨停
                # 向前查找上一个涨停
                for j in range(i-1, max(0, i-30), -1):
                    if is_limit_up[j] == 1:
                        speeds.append(i - j)
                        break
        
        return np.mean(speeds) if speeds else 10  # 默认10天
        
    def calculate_volume_continuity(self, group):
        """计算量能持续性（连续放量天数）"""
        volume_ma5 = group['volume'].rolling(5).mean()
        is_high_volume = (group['volume'] > volume_ma5).astype(int)
        
        # 计算连续天数
        continuity = []
        count = 0
        for val in is_high_volume:
            if val == 1:
                count += 1
            else:
                count = 0
            continuity.append(count)
        
        return continuity
        
    def calculate_volume_pattern(self, group):
        """计算量能形态：递增(1)、递减(-1)、震荡(0)"""
        volume_ma5 = group['volume'].rolling(5).mean()
        volume_ma20 = group['volume'].rolling(20).mean()
        
        # 递增：5日均线持续上穿20日均线
        pattern = []
        for i in range(len(group)):
            if i < 20:  # 数据不足
                pattern.append(0)
            elif (volume_ma5.iloc[i] > volume_ma20.iloc[i]) and \
                 (i > 0 and volume_ma5.iloc[i-1] > volume_ma20.iloc[i-1]):
                pattern.append(1)  # 递增
            elif (volume_ma5.iloc[i] < volume_ma20.iloc[i]) and \
                 (i > 0 and volume_ma5.iloc[i-1] < volume_ma20.iloc[i-1]):
                pattern.append(-1)  # 递减
            else:
                pattern.append(0)  # 震荡
                
        return pattern
        
    def calculate_features_for_stock(self, group):
        """为单只股票计算所有特征"""
        symbol = group['instrument'].iloc[0]
        group = group.sort_values('date').reset_index(drop=True)
        
        # === 预处理：计算is_limit_up ===
        group = self.calculate_limit_up(group)
        
        # === 计算pct_change（如果不存在） ===
        if 'pct_change' not in group.columns:
            group['pct_change'] = (group['close'] - group['close'].shift(1)) / group['close'].shift(1) * 100
        
        # ==================== 股性特征 ====================
        
        # 1. 60日波动率
        group['volatility_60d'] = group['pct_change'].rolling(60).std()
        
        # 2. 60天涨停频率
        group['limit_up_frequency'] = group['is_limit_up'].rolling(60).sum() / 60
        
        # 3. 历史二波成功率（股票级别的常量）
        second_wave_rate = self.get_historical_second_wave_rate(symbol)
        group['second_wave_history'] = second_wave_rate
        
        # 4. 60日平均振幅
        amplitude = (group['high'] - group['low']) / group['close'] * 100
        group['amplitude_avg_60d'] = amplitude.rolling(60).mean()
        
        # 5. 热门股天数（需要hot_rank数据）
        if self.hot_rank_df is not None:
            # 合并热度排名
            group = pd.merge(
                group, 
                self.hot_rank_df[['instrument', 'date', 'rank']].rename(columns={'rank': 'hot_rank'}),
                on=['instrument', 'date'], 
                how='left'
            )
            group['hot_rank'] = group['hot_rank'].fillna(999)
            group['hot_stock_days'] = (group['hot_rank'] <= 100).rolling(60).sum()
        else:
            group['hot_rank'] = 999
            group['hot_stock_days'] = 0
        
        # 6. 概念轮动次数（简化版：统计60天内概念数量的变化）
        if self.concept_df is not None:
            # 统计每个日期该股票所属概念数量
            concept_counts = self.concept_df[
                self.concept_df['member_code'] == symbol
            ].groupby('date').size().reset_index(name='concept_count_daily')
            
            group = pd.merge(
                group,
                concept_counts,
                on='date',
                how='left'
            )
            group['concept_count_daily'] = group['concept_count_daily'].fillna(0)
            
            # 概念轮动次数：60天内概念数量的标准差（波动大=轮动快）
            group['concept_rotation_count'] = group['concept_count_daily'].rolling(60).std().fillna(0)
        else:
            group['concept_rotation_count'] = 0
        
        # 7. 回调反弹速度（整体股票级别）
        rebound_speed = self.calculate_rebound_speed(group)
        group['rebound_speed'] = rebound_speed

        # ==================== 股性偏好增强（A）+ 量价偏好增强（B） ====================
        # 你的偏好：更倾向“有换手、有量一路上来”，且历史上“阳线优势明显 > 阴线”
        #
        # A：阳线优势/趋势质量（60日窗口）
        # - up_day_ratio_60d：上涨天数占比
        # - up_body_sum_ratio_60d：阳线实体总和 / 阴线实体总和
        # - net_body_strength_60d：净实体强度（阳线实体-阴线实体）/ 价格水平
        #
        # B：上涨资金优势（60日窗口）
        # - up_volume_ratio_60d：上涨日成交量 / 下跌日成交量
        # - up_amount_ratio_60d：上涨日成交额 / 下跌日成交额
        # - turnover_trend_20d：换手率趋势（用20日变化率近似）

        eps = 1e-9
        prev_close = group['close'].shift(1)
        up_day = (group['close'] > prev_close)

        # A1: 上涨天数占比
        group['up_day_ratio_60d'] = up_day.astype(int).rolling(60).mean()

        # A2/A3: 阳线/阴线实体优势
        body = (group['close'] - group['open']).fillna(0.0)
        pos_body = body.clip(lower=0.0)
        neg_body = (-body).clip(lower=0.0)
        pos_body_sum_60 = pos_body.rolling(60).sum()
        neg_body_sum_60 = neg_body.rolling(60).sum()
        group['up_body_sum_ratio_60d'] = pos_body_sum_60 / (neg_body_sum_60 + eps)
        # 鲁棒性增强：ratio 可能因分母很小而出现极大值，训练时更推荐用 log1p 版本
        group['up_body_sum_ratio_60d_log'] = np.log1p(group['up_body_sum_ratio_60d'].clip(lower=0.0))

        net_body_sum_60 = body.rolling(60).sum()
        price_level_60 = group['close'].rolling(60).mean()
        group['net_body_strength_60d'] = net_body_sum_60 / (price_level_60 + eps)

        # B1/B2: 上涨日量/额优势
        up_volume_sum_60 = group['volume'].where(up_day, 0).rolling(60).sum()
        down_volume_sum_60 = group['volume'].where(~up_day, 0).rolling(60).sum()
        group['up_volume_ratio_60d'] = up_volume_sum_60 / (down_volume_sum_60 + eps)

        up_amount_sum_60 = group['amount'].where(up_day, 0).rolling(60).sum()
        down_amount_sum_60 = group['amount'].where(~up_day, 0).rolling(60).sum()
        group['up_amount_ratio_60d'] = up_amount_sum_60 / (down_amount_sum_60 + eps)
        
        # ==================== 量价配合特征 ====================
        
        # 8. 20日量价相关性
        group['volume_price_correlation'] = group['volume'].rolling(20).corr(group['close'])
        
        # 9. 放量倍数
        volume_ma20 = group['volume'].rolling(20).mean()
        group['volume_increase_ratio'] = group['volume'] / volume_ma20
        
        # 10. 量价背离
        price_trend = (group['close'] > group['close'].shift(1)).astype(int)
        volume_trend = (group['volume'] > group['volume'].shift(1)).astype(int)
        group['volume_price_divergence'] = (price_trend != volume_trend).astype(int)
        
        # 11. 缩量涨停
        volume_ma5 = group['volume'].rolling(5).mean()
        group['shrink_limit_up'] = (
            (group['is_limit_up'] == 1) & (group['volume'] < volume_ma5)
        ).astype(int)
        
        # 12. 换手率（如果K线数据中有turn字段则用，否则计算）
        if 'turn' in group.columns:
            group['turnover_rate'] = group['turn']
        else:
            # 简化计算：使用volume和amount估算
            # 换手率 = 成交量 / 流通股本 × 100%
            # 这里用 volume / volume_ma60 作为相对换手率的近似
            volume_ma60 = group['volume'].rolling(60).mean()
            group['turnover_rate'] = (group['volume'] / volume_ma60 * 10).fillna(10)  # 归一化到0-100
        
        # 13. 量能持续性
        group['volume_continuity'] = self.calculate_volume_continuity(group)
        
        # 14. 量能形态
        group['volume_pattern'] = self.calculate_volume_pattern(group)
        
        # 15. 量能均线比
        volume_ma5 = group['volume'].rolling(5).mean()
        volume_ma20 = group['volume'].rolling(20).mean()
        group['volume_ma_ratio'] = volume_ma5 / volume_ma20

        # B3: 换手率趋势（20日变化率近似；等下周 turn 落盘后更准确）
        # 说明：这里用 turnover_rate 自身的20日变化率，避免 rolling.apply 过慢
        group['turnover_trend_20d'] = (
            (group['turnover_rate'] / (group['turnover_rate'].shift(20) + eps) - 1)
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=-1.0, upper=10.0)
        )
        
        return group
        
    def calculate_all_features(self):
        """计算所有股票的所有特征"""
        logging.info("开始计算特征...\n")
        
        all_results = []
        stock_list = self.kline_df['instrument'].unique()
        
        for symbol in tqdm(stock_list, desc="计算特征进度"):
            try:
                stock_data = self.kline_df[self.kline_df['instrument'] == symbol].copy()
                stock_with_features = self.calculate_features_for_stock(stock_data)
                all_results.append(stock_with_features)
            except Exception as e:
                logging.error(f"✗ 计算 {symbol} 特征失败: {e}")
                continue
        
        # 合并所有结果
        result_df = pd.concat(all_results, ignore_index=True)
        
        logging.info(f"\n✓ 特征计算完成")
        logging.info(f"  总记录数: {len(result_df)}")
        logging.info(f"  总特征数: {len(result_df.columns)}")
        
        return result_df
        
    def save_features(self, df):
        """保存特征数据"""
        logging.info(f"\n正在保存特征数据到: {self.output_path}")
        
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存CSV
        df.to_csv(self.output_path, index=False)
        
        file_size = self.output_path.stat().st_size / (1024 * 1024)
        logging.info(f"✓ 特征数据已保存")
        logging.info(f"  文件大小: {file_size:.2f} MB")
        logging.info(f"  记录数: {len(df)}")
        logging.info(f"  特征数: {len(df.columns)}")
        
    def print_feature_summary(self, df):
        """打印特征摘要"""
        logging.info("\n" + "=" * 60)
        logging.info("特征摘要")
        logging.info("=" * 60)
        
        new_features = [
            # 股性特征
            'volatility_60d', 'limit_up_frequency', 'second_wave_history',
            'amplitude_avg_60d', 'hot_stock_days', 'concept_rotation_count', 'rebound_speed',
            # 量价特征
            'volume_price_correlation', 'volume_increase_ratio', 'volume_price_divergence',
            'shrink_limit_up', 'turnover_rate', 'volume_continuity', 
            'volume_pattern', 'volume_ma_ratio'
        ]
        
        logging.info("\n[1] 股性特征（7个）:")
        for i, feat in enumerate(new_features[:7], 1):
            if feat in df.columns:
                mean_val = df[feat].mean()
                logging.info(f"  {i}. {feat:25s} : 均值 {mean_val:.4f}")
            else:
                logging.info(f"  {i}. {feat:25s} : [缺失]")
                
        logging.info("\n[2] 量价配合特征（8个）:")
        for i, feat in enumerate(new_features[7:], 1):
            if feat in df.columns:
                mean_val = df[feat].mean()
                logging.info(f"  {i}. {feat:25s} : 均值 {mean_val:.4f}")
            else:
                logging.info(f"  {i}. {feat:25s} : [缺失]")
                
        logging.info("\n" + "=" * 60)
        
    def run(self):
        """运行完整流程"""
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 计算特征
            result_df = self.calculate_all_features()
            
            # 3. 保存特征
            self.save_features(result_df)
            
            # 4. 打印摘要
            self.print_feature_summary(result_df)
            
            logging.info("\n✓ 所有任务完成！")
            return result_df
            
        except Exception as e:
            logging.error(f"\n✗ 执行失败: {e}")
            raise

if __name__ == "__main__":
    engineer = StockCharacterFeatureEngineer()
    result = engineer.run()
