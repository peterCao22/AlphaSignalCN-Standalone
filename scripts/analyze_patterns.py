import os
import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analyze_patterns.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class PatternAnalyzer:
    def __init__(self, processed_dir='data'):
        self.processed_dir = processed_dir
        self.db_path = f"{self.processed_dir}/historical_patterns.db"
        
    def init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建模式表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            trigger_date TEXT,
            trigger_type TEXT,
            consecutive_count INTEGER,
            volume_ratio REAL,
            pre_position TEXT,
            
            -- 后续走势
            max_drawdown REAL,
            max_return_20d REAL,
            second_wave_confirmed INTEGER,
            days_to_second_wave INTEGER,
            
            UNIQUE(symbol, trigger_date)
        )
        """)
        conn.commit()
        conn.close()

    def analyze_events(self):
        """分析所有涨停事件"""
        logging.info("加载涨停事件和K线数据...")
        events_df = pd.read_csv(f"{self.processed_dir}/limit_up_events.csv")
        kline_df = pd.read_csv(f"{self.processed_dir}/kline_processed.csv")
        
        events_df['date'] = pd.to_datetime(events_df['date'])
        kline_df['date'] = pd.to_datetime(kline_df['date'])
        
        conn = sqlite3.connect(self.db_path)
        
        total_events = len(events_df)
        logging.info(f"开始分析 {total_events} 个事件...")
        
        # 按股票分组处理，提高效率
        for instrument, group in kline_df.groupby('instrument'):
            group = group.sort_values('date').reset_index(drop=True)
            instrument_events = events_df[events_df['instrument'] == instrument]
            
            for _, event in instrument_events.iterrows():
                trigger_date = event['date']
                
                # 找到触发日在K线中的索引
                trigger_idx_list = group.index[group['date'] == trigger_date].tolist()
                if not trigger_idx_list:
                    continue
                trigger_idx = trigger_idx_list[0]
                
                # 1. 计算连板数 (consecutive_count)
                consecutive = 1
                check_idx = trigger_idx - 1
                while check_idx >= 0 and group.loc[check_idx, 'is_limit_up']:
                    consecutive += 1
                    check_idx -= 1
                
                # 2. 计算量比 (volume_ratio)
                volume_ratio = group.loc[trigger_idx, 'volume_ratio']
                
                # 3. 计算位置 (pre_position)
                # 简单定义：收盘价 vs MA60
                ma60 = group.loc[trigger_idx, 'ma60']
                pre_position = "低位" if group.loc[trigger_idx, 'close_qfq'] < ma60 else "高位"
                
                # 4. 分析后续走势 (未来30天)
                future_data = group.iloc[trigger_idx+1 : trigger_idx+31]
                if future_data.empty:
                    continue
                
                # 最大回撤 (从触发日收盘价开始算)
                trigger_close = group.loc[trigger_idx, 'close_qfq']
                max_drawdown = (future_data['low_qfq'].min() / trigger_close) - 1
                
                # 20日内最大收益
                max_return_20d = (future_data.iloc[:20]['high_qfq'].max() / trigger_close) - 1
                
                # 二次启动判定 (简化版)
                # 条件：回调后再次突破前高，且涨幅>10%
                second_wave = 0
                days_to_sw = -1
                
                # 找到回调低点
                low_idx = future_data['close_qfq'].idxmin()
                if low_idx > trigger_idx + 1:
                    # 从低点后寻找突破
                    after_low = group.iloc[low_idx+1 : trigger_idx+31]
                    if not after_low.empty:
                        # 突破触发日最高价
                        trigger_high = group.loc[trigger_idx, 'high_qfq']
                        breakthrough = after_low[after_low['close_qfq'] > trigger_high]
                        if not breakthrough.empty:
                            second_wave = 1
                            days_to_sw = (breakthrough.iloc[0]['date'] - trigger_date).days
                
                # 插入数据库
                try:
                    conn.execute("""
                    INSERT OR REPLACE INTO stock_patterns 
                    (symbol, trigger_date, trigger_type, consecutive_count, volume_ratio, pre_position, 
                     max_drawdown, max_return_20d, second_wave_confirmed, days_to_second_wave)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        instrument, trigger_date.strftime('%Y-%m-%d'), '涨停', 
                        int(consecutive), float(volume_ratio) if not np.isnan(volume_ratio) else 0, 
                        pre_position, float(max_drawdown), float(max_return_20d), 
                        int(second_wave), int(days_to_sw)
                    ))
                except Exception as e:
                    logging.error(f"插入失败 {instrument} {trigger_date}: {e}")
            
            # 每处理完一个股票提交一次
            conn.commit()
            
        conn.close()
        logging.info("✓ 模式分析完成，数据库已更新")

if __name__ == "__main__":
    analyzer = PatternAnalyzer()
    analyzer.init_db()
    analyzer.analyze_events()
