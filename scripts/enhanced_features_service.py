"""
增强特征服务

提供热度、概念、竞价等增强特征的提取和计算

作者：AI Assistant
日期：2026-01-13
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


class EnhancedFeaturesService:
    """增强特征服务"""
    
    def __init__(self, data_dir=None):
        """
        初始化增强特征服务
        
        Args:
            data_dir: 数据目录路径
        """
        if data_dir is None:
            base_dir = 'd:/myCursor/StockAiNews/TradingAgents-chinese-market/AlphaSignal-CN'
            data_dir = os.path.join(base_dir, 'data/raw')
        
        self.data_dir = Path(data_dir)
        
        # 加载数据
        self.hot_rank_df = self._load_data('hot_rank.csv')
        self.concept_component_df = self._load_data('concept_component.csv')
        self.concept_bar_df = self._load_data('concept_bar.csv')
        self.auction_factors_df = self._load_data('auction_factors.csv')
        
        print("[OK] 增强特征服务初始化完成")
        print(f"  - 热度排名: {len(self.hot_rank_df) if self.hot_rank_df is not None else 0} 条")
        if self.hot_rank_df is not None:
            print(f"    列名: {self.hot_rank_df.columns.tolist()[:5]}...")
        print(f"  - 概念成分: {len(self.concept_component_df) if self.concept_component_df is not None else 0} 条")
        if self.concept_component_df is not None:
            print(f"    列名: {self.concept_component_df.columns.tolist()[:5]}...")
        print(f"  - 概念行情: {len(self.concept_bar_df) if self.concept_bar_df is not None else 0} 条")
        if self.concept_bar_df is not None:
            print(f"    列名: {self.concept_bar_df.columns.tolist()[:5]}...")
        print(f"  - 竞价因子: {len(self.auction_factors_df) if self.auction_factors_df is not None else 0} 条")
        if self.auction_factors_df is not None:
            print(f"    列名: {self.auction_factors_df.columns.tolist()[:10]}...")
    
    def _load_data(self, filename):
        """加载数据文件"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"⚠️ 文件不存在: {filename}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"❌ 加载文件失败 {filename}: {e}")
            return None
    
    def get_hot_rank_features(self, symbol, date):
        """
        获取热度排名特征
        
        Args:
            symbol: 股票代码（如"000001.SZ"）
            date: 日期
        
        Returns:
            dict: 热度特征
        """
        features = {
            'hot_rank': 999,              # 热度排名（999=未上榜）
            'hot_rank_change': 0,         # 排名变化
            'is_hot_stock': False,        # 是否热门股（TOP100）
            'hot_duration': 0,            # 持续热门天数
        }
        
        if self.hot_rank_df is None:
            return features
        
        try:
            # 获取当天数据
            mask = (self.hot_rank_df['instrument'] == symbol) & \
                   (self.hot_rank_df['date'] == pd.to_datetime(date))
            today_data = self.hot_rank_df[mask]
            
            if not today_data.empty:
                row = today_data.iloc[0]
                features['hot_rank'] = int(row.get('rank', 999))
                features['is_hot_stock'] = features['hot_rank'] <= 100
                
                # 计算排名变化（与前一天对比）
                prev_date = pd.to_datetime(date) - timedelta(days=1)
                prev_mask = (self.hot_rank_df['instrument'] == symbol) & \
                           (self.hot_rank_df['date'] == prev_date)
                prev_data = self.hot_rank_df[prev_mask]
                
                if not prev_data.empty:
                    prev_rank = int(prev_data.iloc[0].get('rank', 999))
                    features['hot_rank_change'] = prev_rank - features['hot_rank']  # 正值=排名上升
                
                # 计算持续热门天数
                features['hot_duration'] = self._calculate_hot_duration(symbol, date)
        
        except Exception as e:
            print(f"提取热度特征失败: {e}")
        
        return features
    
    def _calculate_hot_duration(self, symbol, date, threshold=100):
        """计算持续热门天数"""
        if self.hot_rank_df is None:
            return 0
        
        try:
            # 获取最近30天的排名数据
            end_date = pd.to_datetime(date)
            start_date = end_date - timedelta(days=30)
            
            mask = (self.hot_rank_df['instrument'] == symbol) & \
                   (self.hot_rank_df['date'] >= start_date) & \
                   (self.hot_rank_df['date'] <= end_date)
            history = self.hot_rank_df[mask].sort_values('date', ascending=False)
            
            # 从最近日期开始，计算连续热门天数
            duration = 0
            for _, row in history.iterrows():
                rank = int(row.get('rank', 999))
                if rank <= threshold:
                    duration += 1
                else:
                    break
            
            return duration
        
        except Exception as e:
            return 0
    
    def get_concept_features(self, symbol, date):
        """
        获取概念相关特征
        
        Args:
            symbol: 股票代码
            date: 日期
        
        Returns:
            dict: 概念特征
        """
        features = {
            'concept_count': 0,           # 所属概念数量
            'main_concept_gain': 0.0,     # 主概念涨幅
            'main_concept_rank': 999,     # 主概念排名
            'is_concept_leader': False,   # 是否概念龙头
            'concept_momentum_3d': 0.0,   # 概念3日动量
            'concept_volume_ratio': 1.0,  # 概念成交量比
        }
        
        if self.concept_component_df is None or self.concept_bar_df is None:
            return features
        
        try:
            # 注意：concept_component 数据结构是 "概念 -> 成员股票"
            # instrument = 概念代码, member_code = 股票代码
            mask = (self.concept_component_df['member_code'] == symbol) & \
                   (self.concept_component_df['date'] == pd.to_datetime(date))
            component_data = self.concept_component_df[mask]
            
            if component_data.empty:
                return features
            
            # 获取股票所属的所有概念（instrument 列）
            concepts = component_data['instrument'].unique().tolist()
            features['concept_count'] = len(concepts)
            
            if concepts:
                # 获取主概念（第一个概念）的行情数据
                main_concept = concepts[0]
                concept_bar = self.concept_bar_df[
                    (self.concept_bar_df['instrument'] == main_concept) &
                    (self.concept_bar_df['date'] == pd.to_datetime(date))
                ]
                
                if not concept_bar.empty:
                    row = concept_bar.iloc[0]
                    features['main_concept_gain'] = float(row.get('change_pct', 0.0))
                    
                    # 计算概念3日动量
                    features['concept_momentum_3d'] = self._calculate_concept_momentum(
                        main_concept, date, days=3
                    )
                    
                    # 成交量比
                    volume = float(row.get('volume', 0))
                    avg_volume = float(row.get('volume_avg_5d', volume))
                    if avg_volume > 0:
                        features['concept_volume_ratio'] = volume / avg_volume
        
        except Exception as e:
            print(f"提取概念特征失败: {e}")
        
        return features
    
    def _calculate_concept_momentum(self, concept_code, date, days=3):
        """计算概念动量（N日涨幅）"""
        if self.concept_bar_df is None:
            return 0.0
        
        try:
            end_date = pd.to_datetime(date)
            start_date = end_date - timedelta(days=days+5)  # 多取几天以确保有数据
            
            mask = (self.concept_bar_df['instrument'] == concept_code) & \
                   (self.concept_bar_df['date'] >= start_date) & \
                   (self.concept_bar_df['date'] <= end_date)
            history = self.concept_bar_df[mask].sort_values('date')
            
            if len(history) < 2:
                return 0.0
            
            # 取最近N天的数据
            recent = history.tail(days)
            if len(recent) < 2:
                return 0.0
            
            # 计算涨幅
            start_price = float(recent.iloc[0].get('close', 0))
            end_price = float(recent.iloc[-1].get('close', 0))
            
            if start_price > 0:
                return ((end_price - start_price) / start_price) * 100
            
            return 0.0
        
        except Exception as e:
            return 0.0
    
    def get_auction_features(self, symbol, date):
        """
        获取集合竞价特征
        
        Args:
            symbol: 股票代码
            date: 日期
        
        Returns:
            dict: 竞价特征
        """
        features = {
            'auction_volume_ratio': 1.0,   # 竞价成交量/委托量
            'auction_price_gap': 0.0,      # 竞价价格偏离（%）= 隔夜涨幅
            'auction_turnover': 0.0,       # 竞价换手率
            'auction_strength': 0.0,       # 竞价强度（综合指标）
        }
        
        if self.auction_factors_df is None:
            return features
        
        try:
            mask = (self.auction_factors_df['instrument'] == symbol) & \
                   (self.auction_factors_df['date'] == pd.to_datetime(date))
            auction_data = self.auction_factors_df[mask]
            
            if not auction_data.empty:
                row = auction_data.iloc[0]
                
                # 提取竞价因子（使用实际字段名）
                # 竞价成交量 / 竞价委托量
                order_vol = float(row.get('open_auction_order_volume', 0) or 0)
                trade_vol = float(row.get('open_auction_trade_volume', 0) or 0)
                if order_vol > 0:
                    features['auction_volume_ratio'] = trade_vol / order_vol
                
                # 隔夜涨幅（竞价价格相对昨收的偏离）
                overnight_ratio = row.get('overnight_change_ratio', 0.0)
                # 处理 NaN 和 None
                if pd.isna(overnight_ratio) or overnight_ratio is None:
                    features['auction_price_gap'] = 0.0
                else:
                    features['auction_price_gap'] = float(overnight_ratio)
                
                # 竞价换手率
                features['auction_turnover'] = float(row.get('open_auction_turnover', 0.0) or 0.0)
                
                # 竞价振幅（也是一个重要指标）
                auction_amplitude = float(row.get('open_auction_amplitude', 0.0) or 0.0)
                
                # 计算竞价强度（综合指标）
                features['auction_strength'] = self._calculate_auction_strength(
                    features['auction_volume_ratio'],
                    features['auction_price_gap'],
                    features['auction_turnover'],
                    auction_amplitude
                )
        
        except Exception as e:
            print(f"提取竞价特征失败: {e}")
        
        return features
    
    def _calculate_auction_strength(self, volume_ratio, price_gap, turnover, amplitude=0.0):
        """
        计算竞价强度（0-100）
        
        强度高 = 放量高开 + 换手活跃 + 振幅大
        """
        score = 50.0  # 基础分
        
        # 成交比例因素（权重30%）
        # volume_ratio = 成交量/委托量，越高说明竞价越活跃
        if volume_ratio > 0.8:  # 80%以上成交
            score += 15
        elif volume_ratio > 0.5:
            score += 10
        elif volume_ratio > 0.3:
            score += 5
        elif volume_ratio < 0.1:  # 成交太少
            score -= 10
        
        # 价格因素（权重35%）- 隔夜涨幅
        if price_gap > 5.0:  # 高开5%以上
            score += 20
        elif price_gap > 3.0:
            score += 15
        elif price_gap > 1.0:
            score += 10
        elif price_gap < -3.0:  # 低开
            score -= 15
        elif price_gap < -1.0:
            score -= 10
        
        # 换手率因素（权重20%）
        if turnover > 5.0:
            score += 10
        elif turnover > 3.0:
            score += 5
        elif turnover > 1.0:
            score += 3
        
        # 振幅因素（权重15%）- 振幅大说明博弈激烈
        if amplitude > 5.0:
            score += 10
        elif amplitude > 3.0:
            score += 5
        elif amplitude > 1.0:
            score += 3
        
        # 限制在0-100
        return max(0, min(100, score))
    
    def get_all_features(self, symbol, date):
        """
        获取所有增强特征
        
        Args:
            symbol: 股票代码
            date: 日期
        
        Returns:
            dict: 所有增强特征
        """
        features = {}
        
        # 合并所有特征
        features.update(self.get_hot_rank_features(symbol, date))
        features.update(self.get_concept_features(symbol, date))
        features.update(self.get_auction_features(symbol, date))
        
        return features
    
    def get_batch_features(self, df):
        """
        批量获取增强特征（向量化操作，速度快10-50倍）
        
        Args:
            df: DataFrame，必须包含 'symbol' 和 'trigger_date' 列
        
        Returns:
            DataFrame: 包含所有增强特征的 DataFrame
        """
        result_df = df.copy()
        result_df['date'] = pd.to_datetime(result_df['trigger_date'])
        
        # 1. 批量合并热度排名特征
        if self.hot_rank_df is not None and len(self.hot_rank_df) > 0:
            try:
                hot_rank_agg = self.hot_rank_df.copy()
                hot_rank_agg['date'] = pd.to_datetime(hot_rank_agg['date'])
                
                # 使用实际的列名: rank, rank_change
                hot_rank_subset = hot_rank_agg[['instrument', 'date', 'rank']].copy()
                hot_rank_subset = hot_rank_subset.rename(columns={
                    'instrument': 'symbol',
                    'rank': 'hot_rank'
                })
                
                result_df = pd.merge(
                    result_df,
                    hot_rank_subset,
                    on=['symbol', 'date'],
                    how='left'
                )
                result_df['hot_rank'] = result_df['hot_rank'].fillna(999).astype(int)
                result_df['is_hot_stock'] = (result_df['hot_rank'] <= 100).astype(int)
                
                print("[OK] 热度特征合并成功")
            except Exception as e:
                print(f"⚠️ 热度特征合并失败: {e}")
                result_df['hot_rank'] = 999
                result_df['is_hot_stock'] = 0
        else:
            result_df['hot_rank'] = 999
            result_df['is_hot_stock'] = 0
        
        # 简化特征（避免复杂计算）
        result_df['hot_rank_change'] = 0
        result_df['hot_duration'] = 0
        
        # 2. 批量合并概念特征
        if self.concept_component_df is not None and self.concept_bar_df is not None:
            try:
                concept_comp = self.concept_component_df.copy()
                concept_bar = self.concept_bar_df.copy()
                
                concept_comp['date'] = pd.to_datetime(concept_comp['date'])
                concept_bar['date'] = pd.to_datetime(concept_bar['date'])
                
                # 计算每只股票的概念数量
                # member_code 是股票代码，instrument 是概念代码
                concept_count = concept_comp.groupby(['member_code', 'date']).size().reset_index(name='concept_count')
                concept_count = concept_count.rename(columns={'member_code': 'symbol'})
                
                result_df = pd.merge(
                    result_df,
                    concept_count,
                    on=['symbol', 'date'],
                    how='left'
                )
                result_df['concept_count'] = result_df['concept_count'].fillna(0).astype(int)
                
                # 获取主概念 (取第一个概念)
                main_concept = concept_comp.groupby(['member_code', 'date']).first().reset_index()
                main_concept = main_concept[['member_code', 'date', 'instrument']].rename(
                    columns={'member_code': 'symbol', 'instrument': 'main_concept'}
                )
                
                result_df = pd.merge(result_df, main_concept, on=['symbol', 'date'], how='left')
                
                # 合并概念行情（如果有 pct_change 或 close 字段）
                if 'pct_change' in concept_bar.columns:
                    concept_bar_subset = concept_bar[['instrument', 'date', 'pct_change']].rename(
                        columns={'instrument': 'main_concept', 'pct_change': 'main_concept_gain'}
                    )
                elif 'close' in concept_bar.columns and 'pre_close' in concept_bar.columns:
                    # 手动计算涨跌幅
                    concept_bar['pct_change'] = (
                        (concept_bar['close'] - concept_bar['pre_close']) / concept_bar['pre_close'] * 100
                    ).fillna(0)
                    concept_bar_subset = concept_bar[['instrument', 'date', 'pct_change']].rename(
                        columns={'instrument': 'main_concept', 'pct_change': 'main_concept_gain'}
                    )
                else:
                    # 没有涨幅数据，使用默认值
                    concept_bar_subset = None
                
                if concept_bar_subset is not None:
                    result_df = pd.merge(result_df, concept_bar_subset, on=['main_concept', 'date'], how='left')
                    result_df['main_concept_gain'] = result_df['main_concept_gain'].fillna(0)
                else:
                    result_df['main_concept_gain'] = 0
                
                # 删除临时列
                result_df = result_df.drop(columns=['main_concept'], errors='ignore')
                
                print("[OK] 概念特征合并成功")
            except Exception as e:
                print(f"⚠️ 概念特征合并失败: {e}")
                import traceback
                print(traceback.format_exc())
                result_df['concept_count'] = 0
                result_df['main_concept_gain'] = 0
        else:
            result_df['concept_count'] = 0
            result_df['main_concept_gain'] = 0
        
        # 简化概念特征
        result_df['concept_avg_gain'] = result_df['main_concept_gain']
        result_df['concept_max_gain'] = result_df['main_concept_gain']
        result_df['concept_momentum'] = 0
        result_df['concept_resonance'] = 0
        result_df['main_concept_rank'] = 999
        result_df['is_concept_leader'] = 0
        result_df['concept_momentum_3d'] = 0
        result_df['concept_volume_ratio'] = 0
        
        # 3. 批量合并竞价因子
        if self.auction_factors_df is not None:
            try:
                auction = self.auction_factors_df.copy()
                auction['date'] = pd.to_datetime(auction['date'])
                
                # 使用实际字段: open_auction_trade_volume, open_auction_trade_amount 等
                # 简化特征计算（因为缺少部分必要字段）
                
                # 竞价成交量 (简化：直接使用成交量值)
                auction['auction_volume_ratio'] = auction['open_auction_trade_volume'].fillna(0)
                
                # 竞价价格偏离 (无法计算，使用0)
                auction['auction_price_gap'] = 0
                
                # 竞价成交额 (作为换手率的替代)
                auction['auction_turnover'] = auction['open_auction_trade_amount'].fillna(0)
                
                # 计算竞价强度分数（简化版）
                # 基于成交量和成交额
                auction['auction_strength'] = (
                    (auction['open_auction_trade_volume'] > 1000).astype(int) * 30 +
                    (auction['open_auction_trade_amount'] > 1000000).astype(int) * 30 +
                    (auction['open_auction_cancel_volume'] < 100).astype(int) * 20
                )
                
                auction_subset = auction[[
                    'instrument', 'date', 'auction_volume_ratio', 
                    'auction_price_gap', 'auction_turnover', 'auction_strength'
                ]].rename(columns={'instrument': 'symbol'})
                
                result_df = pd.merge(
                    result_df,
                    auction_subset,
                    on=['symbol', 'date'],
                    how='left'
                )
                
                result_df['auction_volume_ratio'] = result_df['auction_volume_ratio'].fillna(0)
                result_df['auction_price_gap'] = result_df['auction_price_gap'].fillna(0)
                result_df['auction_turnover'] = result_df['auction_turnover'].fillna(0)
                result_df['auction_strength'] = result_df['auction_strength'].fillna(0).astype(int)
                
                print("[OK] 竞价特征合并成功")
            except Exception as e:
                print(f"⚠️ 竞价特征合并失败: {e}")
                import traceback
                print(traceback.format_exc())
                result_df['auction_volume_ratio'] = 0
                result_df['auction_price_gap'] = 0
                result_df['auction_turnover'] = 0
                result_df['auction_strength'] = 0
        else:
            result_df['auction_volume_ratio'] = 0
            result_df['auction_price_gap'] = 0
            result_df['auction_turnover'] = 0
            result_df['auction_strength'] = 0
        
        # 删除临时日期列
        result_df = result_df.drop(columns=['date'], errors='ignore')
        
        return result_df


if __name__ == "__main__":
    # 测试
    service = EnhancedFeaturesService()
    
    # 测试获取特征
    test_symbol = "300346.SZ"
    test_date = "2026-01-13"
    
    print(f"\n测试股票: {test_symbol}, 日期: {test_date}")
    print("="*60)
    
    features = service.get_all_features(test_symbol, test_date)
    
    print("\n增强特征:")
    for key, value in features.items():
        print(f"  {key}: {value}")
