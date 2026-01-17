import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/process_data.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class DataProcessor:
    def __init__(self, raw_dir='data/raw', processed_dir='data/processed'):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def load_kline_data(self):
        """加载全量K线数据"""
        file_path = f"{self.raw_dir}/kline/kline_all.csv"
        if not os.path.exists(file_path):
            logging.error(f"未找到K线数据文件: {file_path}")
            return None
        
        logging.info("加载K线数据...")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def calculate_indicators(self, df):
        """计算技术指标"""
        logging.info("计算技术指标...")
        
        # 按股票代码分组计算
        processed_dfs = []
        
        # 获取最新的复权因子，用于计算前复权 (QFQ)
        # QFQ = HFQ * (current_adjust_factor / latest_adjust_factor)
        # 或者简单点：QFQ = HFQ / latest_adjust_factor (如果我们要让最新价格等于原始价格)
        
        for instrument, group in df.groupby('instrument'):
            group = group.sort_values('date')
            
            # 1. 计算前复权价格 (QFQ)
            # BigQuant 的价格是后复权 (HFQ)
            latest_factor = group['adjust_factor'].iloc[-1]
            for col in ['open', 'high', 'low', 'close']:
                group[f'{col}_qfq'] = group[col] / latest_factor
            
            # 2. 计算涨跌幅
            group['pct_change'] = group['close_qfq'].pct_change() * 100
            
            # 3. 移动平均线 (基于QFQ)
            group['ma5'] = group['close_qfq'].rolling(window=5).mean()
            group['ma10'] = group['close_qfq'].rolling(window=10).mean()
            group['ma20'] = group['close_qfq'].rolling(window=20).mean()
            group['ma60'] = group['close_qfq'].rolling(window=60).mean()
            
            # 4. 成交量均线
            group['v_ma5'] = group['volume'].rolling(window=5).mean()
            group['v_ma20'] = group['volume'].rolling(window=20).mean()
            
            # 5. 量比 (当日成交量 / 过去5日平均成交量)
            group['volume_ratio'] = group['volume'] / group['v_ma5'].shift(1)
            
            # 6. RSI (14)
            delta = group['close_qfq'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            group['rsi'] = 100 - (100 / (1 + rs))
            
            # 7. 标记涨停 (简单判定: 涨幅 > 9.5%)
            # 注意：创业板/科创板是20%，这里简单处理，后续可以根据代码前缀精细化
            limit = 9.5
            if instrument.startswith('30') or instrument.startswith('68'):
                limit = 19.5
            group['is_limit_up'] = group['pct_change'] >= limit
            
            processed_dfs.append(group)
            
        return pd.concat(processed_dfs)

    def process_all(self):
        """执行全流程"""
        df = self.load_kline_data()
        if df is None:
            return
        
        processed_df = self.calculate_indicators(df)
        
        # 保存处理后的数据
        output_file = f"{self.processed_dir}/kline_processed.csv"
        processed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"✓ 数据处理完成，保存至: {output_file}")
        
        # 提取涨停事件库
        limit_ups = processed_df[processed_df['is_limit_up'] == True].copy()
        limit_ups_file = f"{self.processed_dir}/limit_up_events.csv"
        limit_ups.to_csv(limit_ups_file, index=False, encoding='utf-8-sig')
        logging.info(f"✓ 提取涨停事件: {len(limit_ups)} 条，保存至: {limit_ups_file}")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all()
