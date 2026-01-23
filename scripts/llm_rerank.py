"""
LLM二次裁判服务

对 predict_stock.py 的输出结果进行LLM二次裁判，评估股票的位置风险、接力潜力、底部识别和市场环境。

## 功能说明
- 输入：predict_stock.py 生成的预测结果JSON文件（包含股票代码、触发日期、模型预测概率等）
- 输出：增强后的预测结果JSON文件（添加LLM二次裁判结果、投资建议等）
- 核心流程：
  1. 获取K线数据（最近3-6个月，前复权）
  2. 压缩K线数据为文本摘要（识别涨停日、突破日、顶部确认日等关键节点）
  3. 使用web_search获取实时市场信息（股票新闻、市场情绪、板块热点）
  4. 调用LLM进行二次裁判（评估位置风险、接力潜力、底部识别）
  5. 融合模型预测和LLM评分（权重：模型60% + LLM 40%）

## 运行方式
```bash
# 批量处理模式（推荐）
conda run --no-capture-output -n rqsdk python scripts/llm_rerank.py \
  --input results/predictions_trigger_20250115.json \
  --output results/predictions_rerank_20250115.json \
  --topk 20

# 不使用web_search（节省成本，但缺少实时市场信息）
conda run --no-capture-output -n rqsdk python scripts/llm_rerank.py \
  --input results/predictions_trigger_20250115.json \
  --no-web-search
```

## 关键注意事项
1. **数据口径**：
   - K线数据使用前复权（adjust='f'），只使用 ≤ trigger_date 的数据，避免未来信息泄漏
   - 涨停阈值按市场区分：主板10%、创业板/科创板20%、北交所30%
2. **性能与成本**：
   - 默认只对TOP-K（默认20）进行详细LLM裁判，其他快速过滤
   - web_search会消耗API配额，可通过--no-web-search关闭
   - LLM调用使用JSON格式输出，temperature=0.1确保输出稳定
3. **风险点**：
   - 如果LLM响应解析失败，会返回默认结构（rerank_score=0.5, recommendation='hold'）
   - web_search失败时会降级为"未获取到信息"，不影响主流程
   - 股票名称和板块信息目前为TODO，需要后续完善
4. **输出格式**：
   - 在原有prediction_result基础上添加rerank字段
   - 包含rerank_score、risk_level、position_judgment、investment_advice等
   - 最终评分融合：final_score_after_rerank = original_prob * 0.6 + rerank_score * 0.4
"""
import os
import sys
import json
import logging
import argparse
import asyncio
import io
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Windows控制台编码设置（解决乱码问题）
if sys.platform == 'win32':
    # 设置标准输出和错误输出为UTF-8编码
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # 设置控制台代码页为UTF-8
    try:
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True, check=False)
    except:
        pass

# 添加项目根目录到路径
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# 导入本地模块
from stockainews.llm_adapters.doubao_adapter import DoubaoAdapter
from stockainews.services.market_sentiment_service import MarketSentimentService
from stockainews.core.logger import setup_logger
from kline_processing import process_kline

logger = setup_logger(__name__)


class LLMRerankService:
    """
    LLM二次裁判服务类
    
    职责：对模型预测结果进行LLM二次裁判，结合K线形态、实时市场信息评估股票投资价值。
    
    关键属性：
    - llm_adapter: LLM适配器（默认DoubaoAdapter，支持web_search）
    - sentiment_service: 市场情绪服务（用于获取板块信息等）
    - prompt_template: Prompt模板（从stock_rerank_prompt.txt加载）
    - raw_dir: 原始数据目录路径（用于读取本地K线数据）
    """
    
    def __init__(self, llm_adapter=None, raw_dir=None):
        """
        初始化服务
        
        Args:
            llm_adapter: LLM适配器实例（默认使用DoubaoAdapter，支持web_search功能）
            raw_dir: 原始数据目录路径（默认使用REPO_ROOT/data/raw）
        
        注意：
        - 如果未提供适配器，会自动创建默认实例
        - Prompt模板在初始化时加载，如果文件不存在会抛出FileNotFoundError
        - K线数据从本地CSV文件读取，不从API获取
        """
        self.llm_adapter = llm_adapter or DoubaoAdapter()
        self.sentiment_service = MarketSentimentService()
        self.raw_dir = Path(raw_dir) if raw_dir else REPO_ROOT / "data" / "raw"
        self.processed_dir = REPO_ROOT / "data" / "processed"
        self.prompt_template = self._load_prompt_template()
        logger.info("LLM二次裁判服务初始化完成")
    
    def _load_prompt_template(self) -> str:
        """
        加载Prompt模板文件
        
        Returns:
            str: Prompt模板内容（UTF-8编码）
        
        Raises:
            FileNotFoundError: 如果模板文件不存在
        
        注意：
        - 模板文件路径：stockainews/prompts/stock_rerank_prompt.txt
        - 使用UTF-8编码读取，确保中文内容正确显示
        """
        prompt_path = REPO_ROOT / "stockainews" / "prompts" / "stock_rerank_prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt模板文件不存在: {prompt_path}")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _get_limit_up_threshold(self, symbol: str) -> float:
        """
        根据股票代码获取涨停阈值（用于判断涨停/跌停）
        
        Args:
            symbol: 股票代码（如'000001.SZ'或'300001.SZ'，支持带或不带市场后缀）
        
        Returns:
            float: 涨停阈值（0.095=9.5%主板, 0.195=19.5%创业板/科创板, 0.295=29.5%北交所）
        
        注意：
        - 根据股票代码前缀判断市场类型：
          * 300/301开头：创业板（20%涨停）
          * 688/689开头：科创板（20%涨停）
          * 8/43/92开头：北交所（30%涨停）
          * 其他：主板（10%涨停）
        - 返回值为小数形式（如0.095），使用时需乘以100转换为百分比
        """
        # 提取股票代码（去除市场后缀）
        code = symbol.split('.')[0] if '.' in symbol else symbol
        if code.startswith(('300', '301')):
            return 0.195  # 创业板20%
        elif code.startswith(('688', '689')):
            return 0.195  # 科创板20%
        elif code.startswith(('8', '43', '92')):
            return 0.295  # 北交所30%
        else:
            return 0.095  # 主板10%
    
    def _identify_breakthrough_days(
        self, 
        df: pd.DataFrame, 
        ma_period: int = 20,
        pullback_threshold: float = 0.05,
        symbol: str = None
    ) -> List[Dict]:
        """
        识别突破日（放量站上均线→回踩→重新站稳）
        
        突破日定义：前几天放量站上了20日均线，后面有比较深的回踩后重新再次站稳，带有量能。
        这是判断"真底部"和"短期突破"的重要技术形态。
        
        Args:
            df: K线DataFrame（必须已通过process_kline计算MA和volume_ma20等指标）
            ma_period: 均线周期（默认20日，对应MA20）
            pullback_threshold: 回踩阈值（默认5%，即回踩幅度需≥5%）
        
        Returns:
            List[Dict]: 突破日列表，每个元素包含：
                - date: 突破日期（YYYY-MM-DD格式）
                - type: 'breakthrough'
                - description: 突破日描述文本
        
        注意：
        - 需要df包含ma{ma_period}和volume_ma20列，否则返回空列表
        - 向前查找回踩的范围限制在最近10个交易日（避免误判）
        - 放量标准：成交量>2倍均量（volume_ma20）
        - 重新站稳标准：收盘价>均线 且 成交量>均量
        """
        breakthrough_days = []
        ma_col = f'ma{ma_period}'
        volume_ma_col = 'volume_ma20'
        
        # 检查必要的技术指标列是否存在
        if ma_col not in df.columns or volume_ma_col not in df.columns:
            return breakthrough_days
        
        # 从ma_period开始遍历（确保有足够历史数据计算均线）
        # 逻辑：找到"第一次放量站上均线 → 回踩 → 重新站稳"的完整序列
        # 优化：考虑盘中突破（最高价突破均线）和盘中回踩（最低价跌破均线）
        for first_break_idx in range(ma_period, len(df) - 2):  # 第一次站上均线的日期
            # 1. 检查是否第一次放量站上均线
            # 条件1：收盘价>均线 或 最高价>均线（考虑盘中突破）
            # 条件2：成交量>1.5倍均量（放宽放量标准，因为有些突破日可能不是2倍）
            close_above_ma = df.iloc[first_break_idx]['close_qfq'] > df.iloc[first_break_idx][ma_col]
            high_above_ma = df.iloc[first_break_idx].get('high_qfq', df.iloc[first_break_idx]['close_qfq']) > df.iloc[first_break_idx][ma_col]
            volume_above_1_5x = df.iloc[first_break_idx]['volume'] > 1.5 * df.iloc[first_break_idx][volume_ma_col]
            
            # 第一次站上：收盘价或最高价突破均线，且放量（1.5倍均量）
            if (close_above_ma or high_above_ma) and volume_above_1_5x:
                
                # 2. 在第一次站上之后，查找回踩（收盘价或最低价低于均线，且回踩幅度≥pullback_threshold）
                pullback_found = False
                pullback_idx = -1
                pullback_ratio = 0.0
                
                for j in range(first_break_idx + 1, min(len(df), first_break_idx + 20)):  # 在第一次站上之后最多20个交易日内查找回踩
                    # 检查是否有回踩（收盘价或最低价低于均线）
                    close_below_ma = df.iloc[j]['close_qfq'] < df.iloc[j][ma_col]
                    low_below_ma = df.iloc[j].get('low_qfq', df.iloc[j]['close_qfq']) < df.iloc[j][ma_col]
                    
                    if close_below_ma or low_below_ma:
                        # 计算回踩幅度
                        # 如果收盘价低于均线，使用收盘价计算
                        # 如果最低价低于均线但收盘价高于均线（盘中回踩），使用最低价计算
                        if close_below_ma:
                            pullback_price = df.iloc[j]['close_qfq']
                            min_pullback_threshold = pullback_threshold  # 收盘价回踩，使用原阈值（5%）
                        else:
                            pullback_price = df.iloc[j].get('low_qfq', df.iloc[j]['close_qfq'])
                            # 盘中回踩：如果当日跌幅较大（>3%），放宽阈值到0.5%（更宽松）
                            # 否则使用3%阈值
                            pct_change_j = df.iloc[j].get('pct_change', 0) if 'pct_change' in df.columns else 0
                            if pd.notna(pct_change_j) and pct_change_j < -3.0:
                                min_pullback_threshold = 0.005  # 当日跌幅>3%，放宽到0.5%（更宽松，因为盘中回踩幅度可能较小）
                            else:
                                min_pullback_threshold = max(pullback_threshold * 0.6, 0.03)  # 盘中回踩，放宽到3%
                        
                        pullback_ratio = (df.iloc[j][ma_col] - pullback_price) / df.iloc[j][ma_col] if df.iloc[j][ma_col] > 0 else 0
                        # 回踩幅度需≥阈值（收盘价回踩5%，盘中回踩3%或0.5%）
                        if pullback_ratio >= min_pullback_threshold:
                            pullback_found = True
                            pullback_idx = j
                            break
                
                # 3. 如果找到回踩，继续查找重新站稳的日期（在回踩之后）
                if pullback_found:
                    for k in range(pullback_idx + 1, min(len(df), pullback_idx + 10)):  # 在回踩之后最多10个交易日内查找重新站稳
                        # 检查是否重新站稳
                        # 条件1：收盘价>均线（必须）
                        # 条件2：成交量>均量 或 涨停（涨停日即使缩量也算站稳）
                        close_above_ma_k = df.iloc[k]['close_qfq'] > df.iloc[k][ma_col]
                        volume_above_ma_k = df.iloc[k]['volume'] > df.iloc[k][volume_ma_col]
                        # 检查是否涨停（根据symbol获取涨停阈值）
                        is_limit_up_k = False
                        if 'pct_change' in df.columns and symbol:
                            pct_change = df.iloc[k].get('pct_change', 0)
                            if pd.notna(pct_change):
                                limit_up_threshold = self._get_limit_up_threshold(symbol)
                                # 涨幅接近涨停阈值（允许0.5%误差）
                                is_limit_up_k = pct_change >= (limit_up_threshold * 100 - 0.5)
                        
                        if close_above_ma_k and (volume_above_ma_k or is_limit_up_k):
                            # 找到完整的突破序列：第一次站上(first_break_idx) → 回踩(pullback_idx) → 重新站稳(k)
                            breakthrough_days.append({
                                'date': df.iloc[k]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[k]['date'], 'strftime') else str(df.iloc[k]['date']),
                                'type': 'breakthrough',
                                'first_break_date': df.iloc[first_break_idx]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[first_break_idx]['date'], 'strftime') else str(df.iloc[first_break_idx]['date']),
                                'pullback_date': df.iloc[pullback_idx]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[pullback_idx]['date'], 'strftime') else str(df.iloc[pullback_idx]['date']),
                                'pullback_pct': f"{pullback_ratio*100:.2f}%",
                                'description': f"{df.iloc[first_break_idx]['date'].strftime('%Y-%m-%d')}放量站上{ma_period}日均线，{df.iloc[pullback_idx]['date'].strftime('%Y-%m-%d')}回踩{pullback_ratio*100:.2f}%至{ma_period}日线下，{df.iloc[k]['date'].strftime('%Y-%m-%d')}重新站稳{'（涨停）' if is_limit_up_k else '（放量）'}"
                            })
                            break  # 找到一个突破序列就跳出内层循环，继续查找下一个
        
        # 去重：如果同一个重新站稳日期对应多个第一次站上日期，只保留最早的第一次站上日期
        # （因为更早的突破更有意义，表示更早的底部确认）
        if breakthrough_days:
            # 按重新站稳日期分组，每组只保留一个（选择第一次站上日期最早的）
            breakthrough_by_date = {}
            for day in breakthrough_days:
                rerank_date = day['date']
                if rerank_date not in breakthrough_by_date:
                    breakthrough_by_date[rerank_date] = day
                else:
                    # 比较第一次站上日期，保留更早的（因为更早的突破更有意义）
                    existing_first = breakthrough_by_date[rerank_date]['first_break_date']
                    new_first = day['first_break_date']
                    if new_first < existing_first:  # 字符串比较，更早的日期更小
                        breakthrough_by_date[rerank_date] = day
            
            breakthrough_days = list(breakthrough_by_date.values())
        
        return breakthrough_days
    
    def _identify_top_confirmation_days(
        self, 
        df: pd.DataFrame
    ) -> List[Dict]:
        """
        识别顶部确认日（通过跌破关键均线确认顶部）
        
        顶部确认日定义：在回调的过程中，看它是否跌破了短线的三个重要均线：
        - 形态1：破5日线，10日线连续2天不收回。算高点
        - 形态2：破10日线1日不收回，算高点。
        
        这是判断"高位风险"和"顶部派发"的重要技术形态。
        
        Args:
            df: K线DataFrame（必须已通过process_kline计算MA5、MA10等指标）
        
        Returns:
            List[Dict]: 顶部确认日列表，每个元素包含：
                - date: 顶部确认日期（YYYY-MM-DD格式）
                - type: 'top_confirmation'
                - high_date: 对应的高点日期
                - high_price: 高点价格
                - description: 顶部确认描述文本
        
        注意：
        - 需要df包含ma5和ma10列，否则返回空列表
        - 从第10个交易日开始遍历（确保有足够历史数据）
        - 查找高点时向前回溯最多20个交易日
        - 形态1和形态2是互斥的（使用elif），避免重复记录
        """
        top_days = []
        
        # 检查必要的技术指标列是否存在
        if 'ma5' not in df.columns or 'ma10' not in df.columns:
            return top_days
        
        # 从第10个交易日开始遍历（确保有足够历史数据计算均线）
        for i in range(10, len(df)):
            # 形态1：跌破5日线和10日线，连续2天不收回
            # 需要检查i+1是否存在（避免索引越界）
            if (i + 1 < len(df) and 
                df.iloc[i]['close_qfq'] < df.iloc[i]['ma5'] and 
                df.iloc[i]['close_qfq'] < df.iloc[i]['ma10'] and
                df.iloc[i+1]['close_qfq'] < df.iloc[i+1]['ma5'] and 
                df.iloc[i+1]['close_qfq'] < df.iloc[i+1]['ma10']):
                # 查找最近的高点（向前回溯最多20个交易日）
                high_idx = i
                high_price = df.iloc[i]['high_qfq']
                for j in range(max(0, i-20), i):
                    if df.iloc[j]['high_qfq'] > high_price:
                        high_idx = j
                        high_price = df.iloc[j]['high_qfq']
                
                top_days.append({
                    'date': df.iloc[i]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[i]['date'], 'strftime') else str(df.iloc[i]['date']),
                    'type': 'top_confirmation',
                    'high_date': df.iloc[high_idx]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[high_idx]['date'], 'strftime') else str(df.iloc[high_idx]['date']),
                    'high_price': float(high_price),
                    'description': '跌破5日线和10日线，连续2天不收回'
                })
            
            # 形态2：跌破10日线，1天不收回（但未跌破5日线）
            elif (i + 1 < len(df) and 
                  df.iloc[i]['close_qfq'] < df.iloc[i]['ma10'] and
                  df.iloc[i+1]['close_qfq'] < df.iloc[i+1]['ma10'] and
                  df.iloc[i]['close_qfq'] >= df.iloc[i]['ma5']):  # 未跌破5日线（区分形态1和形态2）
                # 查找最近的高点
                high_idx = i
                high_price = df.iloc[i]['high_qfq']
                for j in range(max(0, i-20), i):
                    if df.iloc[j]['high_qfq'] > high_price:
                        high_idx = j
                        high_price = df.iloc[j]['high_qfq']
                
                top_days.append({
                    'date': df.iloc[i]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[i]['date'], 'strftime') else str(df.iloc[i]['date']),
                    'type': 'top_confirmation',
                    'high_date': df.iloc[high_idx]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[high_idx]['date'], 'strftime') else str(df.iloc[high_idx]['date']),
                    'high_price': float(high_price),
                    'description': '跌破10日线，1天不收回'
                })
        
        return top_days
    
    def _compress_kline_data(
        self, 
        kline_data: List[Dict], 
        symbol: str,
        trigger_date: str,
        months: int = 6
    ) -> str:
        """
        压缩和清洗K线数据，生成结构化的文本摘要供LLM分析
        
        将原始K线数据转换为易于LLM理解的文本描述，包括：
        - 基本统计（起始价、最新价、涨跌幅、最高/最低价）
        - 关键节点（涨停日、跌停日、放量日、突破日、顶部确认日）
        - 技术形态（当前价格、均线位置、成交量情况）
        
        Args:
            kline_data: K线数据列表，每个元素包含date、open、high、low、close、volume等字段
            symbol: 股票代码（用于判断交易市场，确定涨停阈值）
            trigger_date: 触发日期（YYYY-MM-DD格式，只使用此日期之前的数据，避免未来信息泄漏）
            months: 保留最近N个月的数据（默认6个月，平衡信息量和token消耗）
        
        Returns:
            str: 压缩后的K线文本描述（Markdown格式）
        
        注意：
        - 数据过滤：先过滤到trigger_date之前，再截取最近N个月（双重过滤确保无未来信息）
        - 涨停判断：使用市场特定的涨停阈值（主板10%、创业板20%、北交所30%），允许0.5%误差
        - 放量标准：成交量>2倍均量（volume_ma20）
        - 关键节点：只显示最近5个涨停/跌停/放量日，最近3个突破/顶部确认日（控制输出长度）
        """
        if not kline_data:
            return "无K线数据"
        
        # 转换为DataFrame（如果kline_data已经是DataFrame格式，直接使用）
        if isinstance(kline_data, pd.DataFrame):
            df = kline_data.copy()
        else:
            df = pd.DataFrame(kline_data)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)
        
        if df.empty:
            return "无有效K线数据"
        
        # 确保有必要的字段（如果从本地加载，应该已经有*_qfq字段）
        if 'close_qfq' not in df.columns:
            # 如果没有*_qfq字段，尝试从原字段创建（假设已是前复权）
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns and f'{col}_qfq' not in df.columns:
                    df[f'{col}_qfq'] = df[col]
        
        # 处理K线数据（计算技术指标：MA5/MA10/MA20、volume_ma20、pct_change等）
        # process_kline会处理复权转换，但我们已经转换过了，所以它会直接使用*_qfq字段
        df = process_kline(df)
        
        # process_kline只计算了v_ma5（5日成交量均线），需要手动计算volume_ma20（20日成交量均线）
        if 'volume' in df.columns:
            # 只在有效交易日上计算（volume > 0）
            valid_mask = df['volume'].fillna(0) > 0
            df['volume_ma20'] = np.nan
            if valid_mask.sum() > 0:
                df_valid = df.loc[valid_mask].copy()
                df_valid['volume_ma20'] = df_valid['volume'].rolling(window=20).mean()
                df.loc[df_valid.index, 'volume_ma20'] = df_valid['volume_ma20']
        
        # 获取涨停阈值（根据市场类型：主板10%、创业板20%、北交所30%）
        limit_up_threshold = self._get_limit_up_threshold(symbol)
        
        # 计算涨跌幅和关键标记
        # 注意：process_kline已经计算了pct_change，但如果数据不足可能为NaN，这里重新计算确保有值
        if 'pct_change' not in df.columns or df['pct_change'].isna().all():
            df['pct_change'] = df['close_qfq'].pct_change() * 100
        
        # 涨停判断：允许0.5%误差（考虑数据精度和四舍五入）
        df['is_limit_up'] = df['pct_change'] >= (limit_up_threshold * 100 - 0.5)
        df['is_limit_down'] = df['pct_change'] <= (-limit_up_threshold * 100 + 0.5)
        # 确定使用的成交量均线列（优先使用volume_ma20，否则使用v_ma5）
        volume_ma_col = 'volume_ma20' if 'volume_ma20' in df.columns else 'v_ma5'
        
        # 放量判断：成交量>2倍均量（如果volume_ma20不存在，使用v_ma5作为备选）
        if volume_ma_col in df.columns:
            df['is_high_volume'] = df['volume'] > 2 * df[volume_ma_col]
        else:
            df['is_high_volume'] = False
        
        # 一字板识别：开盘即涨停，成交量极小（几乎没有换手）
        # 判断标准：
        # 1. 涨停（is_limit_up = True）
        # 2. 开盘价等于或非常接近收盘价（开盘即涨停，允许0.1%误差）
        # 3. 成交量极小（< 0.3倍均量，或 < 平均成交量的30%，或绝对成交量很小）
        df['is_one_word_limit_up'] = False
        # 使用volume_ma20（如果存在），否则使用v_ma5作为备选
        if volume_ma_col in df.columns:
            # 计算平均成交量（用于判断成交量是否极小）
            avg_volume = df['volume'].mean()
            # 计算成交量中位数（更稳健的基准）
            median_volume = df['volume'].median()
            
            for idx in df.index:
                if df.loc[idx, 'is_limit_up']:
                    # 检查价格特征
                    open_price = df.loc[idx, 'open_qfq']
                    close_price = df.loc[idx, 'close_qfq']
                    high_price = df.loc[idx, 'high_qfq']
                    low_price = df.loc[idx, 'low_qfq']
                    
                    # 开盘价接近收盘价（开盘即涨停）
                    price_diff_ratio = abs(open_price - close_price) / close_price if close_price > 0 else 1.0
                    is_price_match = price_diff_ratio < 0.001
                    
                    if not is_price_match:
                        continue
                    
                    # 检查成交量
                    volume = df.loc[idx, 'volume']
                    volume_ma = df.loc[idx, volume_ma_col] if pd.notna(df.loc[idx, volume_ma_col]) else avg_volume
                    volume_ratio = volume / volume_ma if volume_ma > 0 else 1.0
                    
                    # 一字板判断（满足以下任一条件即可）：
                    # 条件a：完全一字板（开=高=低=收，价格完全不动）
                    is_perfect_one_word = (
                        abs(open_price - high_price) / close_price < 0.001 and
                        abs(open_price - low_price) / close_price < 0.001 and
                        abs(high_price - low_price) / close_price < 0.001
                    ) if close_price > 0 else False
                    
                    # 条件b：成交量极小（量比<0.3 或 成交量很小）
                    is_volume_tiny = (
                        volume_ratio < 0.3 or 
                        volume < avg_volume * 0.3 or 
                        volume < median_volume * 0.5 or
                        volume < 10000
                    )
                    
                    # 条件c：准一字板（开盘价=收盘价，且价格波动很小<0.5%）
                    price_amplitude = abs(high_price - low_price) / close_price if close_price > 0 else 0
                    is_quasi_one_word = (price_amplitude < 0.005)  # 波动<0.5%
                    
                    # 满足任一条件即判定为一字板
                    if is_perfect_one_word or is_volume_tiny or is_quasi_one_word:
                        df.loc[idx, 'is_one_word_limit_up'] = True
        
        # 改进方案：代码不判断形态，只提供数据和描述，让LLM自己识别
        # 移除突破日和顶部确认日的计算
        # breakthrough_days = self._identify_breakthrough_days(df, symbol=symbol)  # 移除
        # top_days = self._identify_top_confirmation_days(df)  # 移除
        
        trigger_dt = pd.to_datetime(trigger_date)
        
        # 生成文本摘要
        lines = []
        lines.append(f"## K线数据概览（{df.iloc[0]['date'].strftime('%Y-%m-%d')} 至 {df.iloc[-1]['date'].strftime('%Y-%m-%d')}，共{len(df)}个交易日）")
        lines.append("")
        
        # 基本统计
        start_price = df.iloc[0]['close_qfq']
        latest_price = df.iloc[-1]['close_qfq']
        min_price = df['low_qfq'].min()
        max_price = df['high_qfq'].max()
        min_date = df.loc[df['low_qfq'].idxmin(), 'date']
        max_date = df.loc[df['high_qfq'].idxmax(), 'date']
        avg_volume = df['volume'].mean()
        max_volume = df['volume'].max()
        max_volume_date = df.loc[df['volume'].idxmax(), 'date']
        
        # 计算最大回撤
        max_drawdown = 0.0
        drawdown_date = None
        high_date = None
        if len(df) > 1:
            # 找到最高点
            high_idx = df['high_qfq'].idxmax()
            high_price = df.loc[high_idx, 'high_qfq']
            high_date = df.loc[high_idx, 'date']
            # 在最高点之后找到最低点
            after_high = df.loc[high_idx+1:] if high_idx+1 < len(df) else pd.DataFrame()
            if not after_high.empty:
                low_idx = after_high['low_qfq'].idxmin()
                low_price = after_high.loc[low_idx, 'low_qfq']
                drawdown_date = after_high.loc[low_idx, 'date']
                max_drawdown = ((low_price / high_price) - 1) * 100
        
        lines.append("### 基本统计")
        lines.append("**注意：以下所有价格均为前复权价格**")
        lines.append(f"- 日期范围：{df.iloc[0]['date'].strftime('%Y-%m-%d')} ~ {df.iloc[-1]['date'].strftime('%Y-%m-%d')}（共{len(df)}个交易日）")
        lines.append(f"- 价格区间（前复权）：最低 {min_price:.2f}元（{min_date.strftime('%Y-%m-%d')}），最高 {max_price:.2f}元（{max_date.strftime('%Y-%m-%d')}）")
        lines.append(f"- 当前价格（前复权）：{latest_price:.2f}元（相对最低价+{((latest_price/min_price)-1)*100:.0f}%，相对最高价{((latest_price/max_price)-1)*100:.0f}%）")
        lines.append(f"- 平均成交量：{avg_volume:.0f}，最大成交量：{max_volume:.0f}（{max_volume_date.strftime('%Y-%m-%d')}）")
        if max_drawdown < 0 and drawdown_date and high_date:
            lines.append(f"- 最大回撤：{max_drawdown:.2f}%（{drawdown_date.strftime('%Y-%m-%d')}相对{high_date.strftime('%Y-%m-%d')}高点）")
        lines.append("")
        
        # 一字板统计信息（改进：只统计触发日及其前5个交易日内的一字板，不判断风险）
        # 找到触发日在df中的位置
        trigger_idx_list = df[df['date'] == trigger_dt].index
        if len(trigger_idx_list) > 0:
            trigger_idx = trigger_idx_list[0]
            # 获取触发日及其前5个交易日的数据（共6天，包含触发日当天）
            start_idx = max(0, trigger_idx - 5)
            end_idx = trigger_idx + 1  # +1 因为切片不包含结束索引
            df_recent = df.iloc[start_idx:end_idx].copy()
            one_word_days = df_recent[df_recent['is_one_word_limit_up']].copy()
        else:
            # 触发日不在df中，检查最后6天
            one_word_days = df.tail(6)[df.tail(6)['is_one_word_limit_up']].copy()
        
        if not one_word_days.empty:
            lines.append("### 一字板统计信息")
            lines.append(f"- 统计范围：触发日（{trigger_dt.strftime('%Y-%m-%d')}）及其前5个交易日（共6天）")
            lines.append(f"- 出现次数：{len(one_word_days)}次")
            
            # 计算每个一字板的信息
            one_word_info = []
            for _, row in one_word_days.iterrows():
                one_word_date = row['date']
                volume_ma_val = row.get(volume_ma_col, row.get('v_ma5', 1))
                volume_ratio = row['volume'] / volume_ma_val if pd.notna(volume_ma_val) and volume_ma_val > 0 else 0
                
                # 计算距离触发日期的天数
                days_before_trigger = (trigger_dt - one_word_date).days if trigger_dt >= one_word_date else 0
                
                # 判断位置（高位/低位，基于MA60）
                position = "高位" if pd.notna(row.get('ma60')) and row['close_qfq'] > row['ma60'] else "低位"
                
                # 获取该一字板日期前5个交易日的K线信息
                one_word_idx = df[df['date'] == one_word_date].index
                before_5d_limit_up_count = 0
                before_5d_price_range = "N/A"
                before_5d_avg_volume = 0
                
                if len(one_word_idx) > 0:
                    idx = one_word_idx[0]
                    prev_5d_start = max(0, idx - 5)
                    prev_5d_data = df.iloc[prev_5d_start:idx]
                    before_5d_limit_up_count = prev_5d_data['is_limit_up'].sum()
                    before_5d_avg_volume = prev_5d_data['volume'].mean() if not prev_5d_data.empty else 0
                    before_5d_price_range = f"{prev_5d_data['close_qfq'].min():.2f}-{prev_5d_data['close_qfq'].max():.2f}" if not prev_5d_data.empty else "N/A"
                
                one_word_info.append({
                    'date': one_word_date.strftime('%Y-%m-%d'),
                    'position': position,
                    'days_before_trigger': days_before_trigger,
                    'volume_ratio': volume_ratio,
                    'before_5d_limit_up_count': before_5d_limit_up_count,
                    'before_5d_price_range': before_5d_price_range,
                    'before_5d_avg_volume': before_5d_avg_volume
                })
            
            # 按日期排序
            one_word_info.sort(key=lambda x: x['date'])
            
            # 生成统计信息文本
            dates_str = ", ".join([info['date'] for info in one_word_info])
            positions_str = ", ".join([f"{info['position']}（{info['date']}）" for info in one_word_info])
            days_before_str = ", ".join([f"{info['days_before_trigger']}天前" for info in one_word_info])
            volume_ratios_str = ", ".join([f"{info['volume_ratio']:.2f}（{info['date']}）" for info in one_word_info])
            
            lines.append(f"- 出现日期：{dates_str}")
            lines.append(f"- 出现位置：{positions_str}")
            lines.append(f"- 距离触发日期：{days_before_str}")
            lines.append(f"- 量比：{volume_ratios_str}")
            lines.append("- 上下文信息：")
            for info in one_word_info:
                lines.append(f"  - {info['date']}前5个交易日：涨停{info['before_5d_limit_up_count']}次，价格区间{info['before_5d_price_range']}元，平均成交量{info['before_5d_avg_volume']:.0f}")
            lines.append("")
        
        # 关键节点（按时间顺序，包含涨停、跌停、放量等重要事件，但不判断形态）
        limit_up_days = df[df['is_limit_up']].copy()
        limit_down_days = df[df['is_limit_down']].copy()
        high_volume_days = df[df['is_high_volume']].copy()
        
        # 合并所有关键节点，按时间排序
        key_nodes = []
        
        # 涨停日（排除一字板，因为一字板已经在上面单独统计了）
        normal_limit_up_days = limit_up_days[~limit_up_days['is_one_word_limit_up']].copy()
        for _, row in normal_limit_up_days.iterrows():
            key_nodes.append({
                'date': row['date'],
                'type': '涨停',
                'description': f"涨停，收盘价（前复权）{row['close_qfq']:.2f}元，涨幅{row['pct_change']:.2f}%，成交量{row['volume']:.0f}"
            })
        
        # 跌停日
        for _, row in limit_down_days.iterrows():
            key_nodes.append({
                'date': row['date'],
                'type': '跌停',
                'description': f"跌停，收盘价（前复权）{row['close_qfq']:.2f}元，跌幅{row['pct_change']:.2f}%"
            })
        
        # 放量日
        for _, row in high_volume_days.iterrows():
            volume_ma_val = row.get(volume_ma_col, row.get('v_ma5', 1))
            volume_ratio = row['volume'] / volume_ma_val if pd.notna(volume_ma_val) and volume_ma_val > 0 else 0
            key_nodes.append({
                'date': row['date'],
                'type': '放量',
                'description': f"放量，成交量{row['volume']:.0f}，量比{volume_ratio:.2f}（成交量>2倍均量）"
            })
        
        # 按日期排序，只显示最近的关键节点（控制输出长度）
        key_nodes.sort(key=lambda x: x['date'])
        
        if key_nodes:
            lines.append("### 关键节点（按时间顺序，最近10个重要事件）")
            for node in key_nodes[-10:]:
                lines.append(f"- {node['date'].strftime('%Y-%m-%d')}：{node['description']}")
            lines.append("")
        
        # 价格走势描述（关键区间，便于LLM理解整体趋势和识别形态）
        # 识别关键区间：涨停日、跌停日、放量日、价格大幅波动等
        lines.append("### 价格走势描述（关键区间）")
        
        # 将K线数据按关键事件分段描述
        # 简化：每10-20个交易日为一个区间，描述价格变化和成交量情况
        interval_size = 15  # 每15个交易日为一个区间
        for i in range(0, len(df), interval_size):
            interval_df = df.iloc[i:min(i+interval_size, len(df))]
            if len(interval_df) < 2:
                continue
            
            start_date = interval_df.iloc[0]['date']
            end_date = interval_df.iloc[-1]['date']
            start_price_interval = interval_df.iloc[0]['close_qfq']
            end_price_interval = interval_df.iloc[-1]['close_qfq']
            pct_change_interval = ((end_price_interval / start_price_interval) - 1) * 100
            
            # 判断成交量变化
            avg_volume_start = interval_df.iloc[:len(interval_df)//2]['volume'].mean()
            avg_volume_end = interval_df.iloc[len(interval_df)//2:]['volume'].mean()
            volume_trend = "放大" if avg_volume_end > avg_volume_start * 1.2 else ("萎缩" if avg_volume_end < avg_volume_start * 0.8 else "平稳")
            
            # 判断是否站上/跌破均线（使用区间结束时的MA20）
            ma20_end = interval_df.iloc[-1].get('ma20')
            price_vs_ma20 = ""
            if pd.notna(ma20_end) and ma20_end > 0:
                if end_price_interval > ma20_end:
                    price_vs_ma20 = "，站上MA20"
                elif end_price_interval < ma20_end:
                    price_vs_ma20 = "，跌破MA20"
            
            # 判断涨跌
            trend = "上涨" if pct_change_interval > 0 else "下跌"
            
            lines.append(f"- {start_date.strftime('%Y-%m-%d')}至{end_date.strftime('%Y-%m-%d')}：从{start_price_interval:.2f}元（前复权）{trend}至{end_price_interval:.2f}元（前复权，{pct_change_interval:+.2f}%），成交量{volume_trend}{price_vs_ma20}")
        
        lines.append("")
        
        # 技术指标（最近N个交易日的关键数据，便于LLM判断形态）
        recent_n = 10  # 最近10个交易日
        if len(df) >= recent_n:
            lines.append(f"### 技术指标（最近{recent_n}个交易日）")
            recent_df = df.iloc[-recent_n:]
            
            # MA5, MA10, MA20, MA60
            ma5_values = ", ".join([f"{v:.2f}" if pd.notna(v) else "N/A" for v in recent_df['ma5'].tail(5)])
            ma10_values = ", ".join([f"{v:.2f}" if pd.notna(v) else "N/A" for v in recent_df['ma10'].tail(5)])
            ma20_values = ", ".join([f"{v:.2f}" if pd.notna(v) else "N/A" for v in recent_df['ma20'].tail(5)])
            ma60_values = ", ".join([f"{v:.2f}" if pd.notna(v) else "N/A" for v in recent_df['ma60'].tail(5)])
            
            lines.append(f"- MA5（最近5个交易日，前复权）: {ma5_values}")
            lines.append(f"- MA10（最近5个交易日，前复权）: {ma10_values}")
            lines.append(f"- MA20（最近5个交易日，前复权）: {ma20_values}")
            if pd.notna(recent_df['ma60'].iloc[-1]):
                lines.append(f"- MA60（最近5个交易日，前复权）: {ma60_values}")
            
            # 成交量
            volume_values = ", ".join([f"{v:.0f}" for v in recent_df['volume'].tail(5)])
            lines.append(f"- 成交量（最近5个交易日）: {volume_values}")
            
            # 量比
            volume_ratio_values = []
            for _, row in recent_df.tail(5).iterrows():
                volume_ma_val = row.get(volume_ma_col, row.get('v_ma5', 1))
                volume_ratio = row['volume'] / volume_ma_val if pd.notna(volume_ma_val) and volume_ma_val > 0 else 0
                volume_ratio_values.append(f"{volume_ratio:.2f}")
            lines.append(f"- 量比（最近5个交易日）: {', '.join(volume_ratio_values)}")
            
            # RSI
            if 'rsi' in recent_df.columns:
                rsi_values = ", ".join([f"{v:.2f}" if pd.notna(v) else "N/A" for v in recent_df['rsi'].tail(5)])
                lines.append(f"- RSI（最近5个交易日）: {rsi_values}")
            
            lines.append("")
        
        # 关键事件标记（汇总）
        lines.append("### 关键事件标记（汇总）")
        limit_up_dates_str = ", ".join([d.strftime('%Y-%m-%d') for d in limit_up_days['date'].tail(10)])
        limit_down_dates_str = ", ".join([d.strftime('%Y-%m-%d') for d in limit_down_days['date'].tail(10)])
        high_volume_dates_str = ", ".join([d.strftime('%Y-%m-%d') for d in high_volume_days['date'].tail(10)])
        
        lines.append(f"- 涨停日：{limit_up_dates_str}（共{len(limit_up_days)}次）")
        lines.append(f"- 跌停日：{limit_down_dates_str}（共{len(limit_down_days)}次）")
        lines.append(f"- 放量日：{high_volume_dates_str}（成交量>2倍均量，共{len(high_volume_days)}次）")
        lines.append("")
        
        # 当前技术形态
        lines.append("### 当前技术形态")
        latest = df.iloc[-1]
        lines.append(f"- 当前价格（前复权）: {latest['close_qfq']:.2f}元")
        ma_line = f"- MA5（前复权）: {latest['ma5']:.2f}元，MA10（前复权）: {latest['ma10']:.2f}元，MA20（前复权）: {latest['ma20']:.2f}元"
        if pd.notna(latest.get('ma60')):
            ma_line += f"，MA60（前复权）: {latest['ma60']:.2f}元"
        lines.append(ma_line)
        
        if pd.notna(latest['ma20']) and latest['ma20'] > 0:
            lines.append(f"- 价格相对MA20位置: {((latest['close_qfq'] / latest['ma20']) - 1) * 100:.2f}%")
        volume_ma_val = latest.get(volume_ma_col, latest.get('v_ma5', 1))
        volume_ratio = latest['volume'] / volume_ma_val if pd.notna(volume_ma_val) and volume_ma_val > 0 else 0
        lines.append(f"- 成交量: {latest['volume']:.0f}，量比: {volume_ratio:.2f}")
        if pd.notna(latest.get('rsi')):
            lines.append(f"- RSI: {latest['rsi']:.2f}")
        
        return "\n".join(lines)
    
    def _extract_one_word_stats(
        self,
        df: pd.DataFrame,
        symbol: str,
        trigger_date: str
    ) -> str:
        """
        提取一字板统计信息（用于单独传递给LLM）
        
        改进方案：代码只统计信息，不判断风险，让LLM根据统计信息判断风险
        
        注意：只统计触发日及其前5个交易日内的一字板（共6天），超出范围的一字板不统计
        
        Args:
            df: K线DataFrame（必须已通过process_kline计算技术指标）
            symbol: 股票代码
            trigger_date: 触发日期
        
        Returns:
            str: 一字板统计信息文本（Markdown格式）
        """
        if df.empty or 'is_one_word_limit_up' not in df.columns:
            return "### 一字板统计信息\n- 出现次数：0次（统计范围：触发日及其前5个交易日）"
        
        trigger_dt = pd.to_datetime(trigger_date)
        
        # 只统计触发日及其前5个交易日内的一字板（共6天）
        # 找到触发日在df中的位置
        trigger_idx = df[df['date'] == trigger_dt].index
        if len(trigger_idx) == 0:
            # 触发日不在df中，使用最后一天
            trigger_idx = [len(df) - 1]
        else:
            trigger_idx = [trigger_idx[0]]
        
        # 获取触发日及其前5个交易日的数据（共6天，包含触发日当天）
        start_idx = max(0, trigger_idx[0] - 5)
        end_idx = trigger_idx[0] + 1  # +1 因为切片不包含结束索引
        df_recent = df.iloc[start_idx:end_idx].copy()
        
        one_word_days = df_recent[df_recent['is_one_word_limit_up']].copy()
        if one_word_days.empty:
            return "### 一字板统计信息\n- 出现次数：0次（统计范围：触发日及其前5个交易日）"
        
        trigger_dt = pd.to_datetime(trigger_date)
        volume_ma_col = 'volume_ma20' if 'volume_ma20' in df.columns else 'v_ma5'
        
        lines = []
        lines.append("### 一字板统计信息")
        lines.append(f"- 统计范围：触发日（{trigger_dt.strftime('%Y-%m-%d')}）及其前5个交易日（共6天）")
        lines.append(f"- 出现次数：{len(one_word_days)}次")
        
        # 计算每个一字板的信息
        one_word_info = []
        for _, row in one_word_days.iterrows():
            one_word_date = row['date']
            volume_ma_val = row.get(volume_ma_col, row.get('v_ma5', 1))
            volume_ratio = row['volume'] / volume_ma_val if pd.notna(volume_ma_val) and volume_ma_val > 0 else 0
            
            # 计算距离触发日期的天数
            days_before_trigger = (trigger_dt - one_word_date).days if trigger_dt >= one_word_date else 0
            
            # 判断位置（高位/低位，基于MA60）
            position = "高位" if pd.notna(row.get('ma60')) and row['close_qfq'] > row['ma60'] else "低位"
            
            # 获取该一字板日期前5个交易日的K线信息
            one_word_idx = df[df['date'] == one_word_date].index
            before_5d_limit_up_count = 0
            before_5d_price_range = "N/A"
            before_5d_avg_volume = 0
            
            if len(one_word_idx) > 0:
                idx = one_word_idx[0]
                prev_5d_start = max(0, idx - 5)
                prev_5d_data = df.iloc[prev_5d_start:idx]
                before_5d_limit_up_count = prev_5d_data['is_limit_up'].sum()
                before_5d_avg_volume = prev_5d_data['volume'].mean() if not prev_5d_data.empty else 0
                before_5d_price_range = f"{prev_5d_data['close_qfq'].min():.2f}-{prev_5d_data['close_qfq'].max():.2f}" if not prev_5d_data.empty else "N/A"
            
            one_word_info.append({
                'date': one_word_date.strftime('%Y-%m-%d'),
                'position': position,
                'days_before_trigger': days_before_trigger,
                'volume_ratio': volume_ratio,
                'before_5d_limit_up_count': before_5d_limit_up_count,
                'before_5d_price_range': before_5d_price_range,
                'before_5d_avg_volume': before_5d_avg_volume
            })
        
        # 按日期排序
        one_word_info.sort(key=lambda x: x['date'])
        
        # 生成统计信息文本
        dates_str = ", ".join([info['date'] for info in one_word_info])
        positions_str = ", ".join([f"{info['position']}（{info['date']}）" for info in one_word_info])
        days_before_str = ", ".join([f"{info['days_before_trigger']}天前" for info in one_word_info])
        volume_ratios_str = ", ".join([f"{info['volume_ratio']:.2f}（{info['date']}）" for info in one_word_info])
        
        lines.append(f"- 出现日期：{dates_str}")
        lines.append(f"- 出现位置：{positions_str}")
        lines.append(f"- 距离触发日期：{days_before_str}")
        lines.append(f"- 量比：{volume_ratios_str}")
        lines.append("- 上下文信息：")
        for info in one_word_info:
            lines.append(f"  - {info['date']}前5个交易日：涨停{info['before_5d_limit_up_count']}次，价格区间{info['before_5d_price_range']}元，平均成交量{info['before_5d_avg_volume']:.0f}")
        
        return "\n".join(lines)
    
    async def _web_search_market_info(
        self, 
        symbol: str, 
        stock_name: str, 
        trigger_date: str,
        sector_name: str = None
    ) -> Dict[str, str]:
        """
        使用豆包web_search搜索实时市场信息
        
        通过web_search获取触发日期所在月份的市场信息，包括：
        - 股票相关新闻（如果提供股票名称）
        - 市场情绪和热点（A股整体情况）
        - 板块/概念炒作情况（如果提供板块名称）
        
        Args:
            symbol: 股票代码（用于搜索查询）
            stock_name: 股票名称（如果为空则跳过股票新闻搜索）
            trigger_date: 触发日期（YYYY-MM-DD格式，用于提取年月构建搜索词）
            sector_name: 板块名称（可选，如果为空则跳过板块热点搜索）
        
        Returns:
            Dict[str, str]: 搜索结果字典，包含：
                - stock_news: 股票相关新闻摘要（如果搜索失败则返回"未获取到股票相关新闻"）
                - market_sentiment: 市场情绪搜索结果（如果搜索失败则返回"未获取到市场情绪信息"）
                - sector_hotspot: 板块热点搜索结果（如果未提供板块名称则返回"未提供板块信息"）
        
        注意：
        - 搜索词中明确包含年月（如"2025年1月"），避免获取未来信息
        - 每个搜索都有独立的异常处理，单个搜索失败不影响其他搜索
        - web_search会消耗API配额，失败时会降级为默认文本，不影响主流程
        - 搜索是异步的，需要在async函数中调用
        """
        results = {
            'stock_news': '',
            'market_sentiment': '',
            'sector_hotspot': ''
        }
        
        try:
            # 提取年月用于搜索（明确日期范围，避免获取未来信息）
            trigger_dt = pd.to_datetime(trigger_date)
            year_month = trigger_dt.strftime('%Y年%m月')
            
            # 搜索股票相关新闻（如果提供股票名称）
            if stock_name:
                query1 = f"{year_month} {stock_name}({symbol}) 股票新闻 市场动态"
                try:
                    results['stock_news'] = await self.llm_adapter.web_search(query1)
                except Exception as e:
                    logger.warning(f"搜索股票新闻失败: {e}")
                    results['stock_news'] = "未获取到股票相关新闻"  # 降级处理
            
            # 搜索市场情绪（A股整体情况）
            query2 = f"{year_month} A股市场情绪 涨停数量 市场热点"
            try:
                results['market_sentiment'] = await self.llm_adapter.web_search(query2)
            except Exception as e:
                logger.warning(f"搜索市场情绪失败: {e}")
                results['market_sentiment'] = "未获取到市场情绪信息"  # 降级处理
            
            # 搜索板块热点（如果提供板块名称）
            if sector_name:
                query3 = f"{year_month} {sector_name} 板块热点 概念炒作"
                try:
                    results['sector_hotspot'] = await self.llm_adapter.web_search(query3)
                except Exception as e:
                    logger.warning(f"搜索板块热点失败: {e}")
                    results['sector_hotspot'] = "未获取到板块热点信息"  # 降级处理
            else:
                results['sector_hotspot'] = "未提供板块信息"
        
        except Exception as e:
            logger.error(f"Web搜索过程出错: {e}", exc_info=True)
            # 外层异常处理：确保即使整体失败也返回默认结构
        
        return results
    
    def _load_local_kline_data(
        self,
        symbol: str,
        trigger_date: str,
        kline_months: int = 6
    ) -> pd.DataFrame:
        """
        从本地CSV文件加载K线数据
        
        数据来源优先级：
        1. data/processed/kline_processed.csv（补充后的最新数据）
        2. data/raw/kline/kline_all.csv（原始数据）
        
        Args:
            symbol: 股票代码（如'000001.SZ'）
            trigger_date: 触发日期（YYYY-MM-DD格式）
            kline_months: 保留最近N个月的数据
        
        Returns:
            pd.DataFrame: K线数据DataFrame，包含date、instrument、open、high、low、close、volume等字段
        
        注意：
        - 本地K线数据是后复权（HFQ），需要转换为前复权（QFQ）用于技术分析
        - 如果有adjust_factor字段，使用它进行转换：QFQ = HFQ / latest_factor
        - 如果没有adjust_factor，假设数据已经是前复权（补充数据可能是前复权）
        """
        # 提取股票代码（去除市场后缀）
        stock_code = symbol.split('.')[0] if '.' in symbol else symbol
        
        # 优先读取processed目录（补充后的最新数据）
        kline_paths = [
            self.processed_dir / "kline_processed.csv",
            self.raw_dir / "kline" / "kline_all.csv"
        ]
        
        df_kline = None
        for kline_path in kline_paths:
            if kline_path.exists():
                try:
                    logger.info(f"正在从本地读取K线数据: {kline_path}")
                    df_kline_all = pd.read_csv(kline_path, dtype={'stock_code': str})
                    df_kline = df_kline_all[df_kline_all['instrument'] == symbol].copy()
                    if not df_kline.empty:
                        logger.info(f"成功加载K线数据: {len(df_kline)} 条记录")
                        break
                except Exception as e:
                    logger.warning(f"读取 {kline_path} 失败: {e}")
                    continue
        
        if df_kline is None or df_kline.empty:
            logger.warning(f"未找到 {symbol} 的K线数据")
            return pd.DataFrame()
        
        # 标准化日期格式
        df_kline['date'] = pd.to_datetime(df_kline['date'], format='ISO8601', errors='coerce')
        df_kline = df_kline.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
        
        # 过滤到trigger_date之前（避免未来信息泄漏）
        trigger_dt = pd.to_datetime(trigger_date)
        df_kline = df_kline[df_kline['date'] <= trigger_dt].copy()
        
        # 只保留最近N个月
        cutoff_date = trigger_dt - pd.DateOffset(months=kline_months)
        df_kline = df_kline[df_kline['date'] >= cutoff_date].copy()
        
        if df_kline.empty:
            logger.warning(f"过滤后无有效K线数据（{symbol}, {trigger_date}）")
            return pd.DataFrame()
        
        # 复权转换：本地数据是后复权（HFQ），需要转换为前复权（QFQ）
        # 如果有adjust_factor字段，使用它进行转换
        if 'adjust_factor' in df_kline.columns:
            # 获取最新的复权因子（用于前复权转换）
            latest_factor = df_kline['adjust_factor'].iloc[-1]
            if pd.notna(latest_factor) and latest_factor > 0:
                # 前复权转换：QFQ = HFQ / latest_factor
                for col in ['open', 'high', 'low', 'close']:
                    if col in df_kline.columns:
                        df_kline[f'{col}_qfq'] = df_kline[col] / latest_factor
                logger.info(f"已转换为前复权（使用adjust_factor，latest_factor={latest_factor:.6f}）")
            else:
                # adjust_factor无效，直接使用原价（假设已是前复权）
                for col in ['open', 'high', 'low', 'close']:
                    if col in df_kline.columns:
                        df_kline[f'{col}_qfq'] = df_kline[col]
                logger.warning(f"adjust_factor无效，直接使用原价（假设已是前复权）")
        else:
            # 没有adjust_factor字段，假设数据已经是前复权（补充数据可能是前复权）
            for col in ['open', 'high', 'low', 'close']:
                if col in df_kline.columns:
                    df_kline[f'{col}_qfq'] = df_kline[col]
            logger.info("未找到adjust_factor字段，直接使用原价（假设已是前复权）")
        
        return df_kline
    
    def _prepare_context(
        self, 
        prediction_result: Dict, 
        use_web_search: bool = True,
        kline_months: int = 6
    ) -> Dict:
        """
        准备LLM二次裁判所需的上下文数据
        
        从prediction_result中提取信息，从本地CSV文件读取K线数据并压缩，提取模型预测结果。
        注意：web_search是异步的，需要在外部调用时await。
        
        Args:
            prediction_result: predict_stock.py 的输出结果，必须包含：
                - symbol: 股票代码
                - trigger_date: 触发日期
                - features: 特征字典（包含consecutive_count、pre_position等）
                - scores: 评分字典（包含final_prob、pattern_success_rate等）
                - conclusion: 模型结论
            use_web_search: 是否使用web_search获取实时信息（默认True，但实际搜索在外部await）
            kline_months: K线数据保留最近N个月（默认6个月）
        
        Returns:
            Dict: 上下文字典，包含：
                - symbol: 股票代码
                - stock_name: 股票名称（目前为TODO，返回空字符串）
                - trigger_date: 触发日期
                - kline_summary: 压缩后的K线文本摘要
                - model_summary: 模型预测结果摘要
                - use_web_search: 是否使用web_search（用于外部判断）
                - sector_name: 板块名称（目前为TODO，返回None）
        
        注意：
        - K线数据从本地CSV文件读取（data/processed/kline_processed.csv 或 data/raw/kline/kline_all.csv）
        - 本地数据是后复权（HFQ），会自动转换为前复权（QFQ）用于技术分析
        - K线数据获取失败时会记录警告，但继续执行（kline_summary可能为空）
        - 股票名称和板块信息目前为TODO，需要后续从数据库或其他来源获取
        - web_search结果需要在外部异步获取后添加到context中
        """
        symbol = prediction_result.get('symbol', '')
        trigger_date = prediction_result.get('trigger_date', '')
        
        # 从本地CSV文件加载K线数据（后复权转前复权）
        df_kline = self._load_local_kline_data(symbol, trigger_date, kline_months=kline_months)
        
        # 转换为字典列表格式（用于_compress_kline_data）
        if not df_kline.empty:
            kline_data = df_kline.to_dict('records')
        else:
            kline_data = []
        
        # 先处理K线数据（计算技术指标），用于获取当前价格和均线，以及提取一字板统计信息
        current_price_qfq = None
        current_ma5 = None
        current_ma10 = None
        current_ma20 = None
        current_ma60 = None
        
        if not df_kline.empty:
            df_processed = process_kline(df_kline.copy())
            # 计算volume_ma20
            if 'volume' in df_processed.columns:
                valid_mask = df_processed['volume'].fillna(0) > 0
                df_processed['volume_ma20'] = np.nan
                if valid_mask.sum() > 0:
                    df_valid = df_processed.loc[valid_mask].copy()
                    df_valid['volume_ma20'] = df_valid['volume'].rolling(window=20).mean()
                    df_processed.loc[df_valid.index, 'volume_ma20'] = df_valid['volume_ma20']
            
            # 获取触发日期的当前价格和均线（前复权）- 用于在Prompt中明确提供，确保LLM使用正确价格
            trigger_dt = pd.to_datetime(trigger_date)
            trigger_day_data = df_processed[df_processed['date'] == trigger_dt]
            if not trigger_day_data.empty:
                trigger_row = trigger_day_data.iloc[0]
                current_price_qfq = trigger_row.get('close_qfq')
                current_ma5 = trigger_row.get('ma5')
                current_ma10 = trigger_row.get('ma10')
                current_ma20 = trigger_row.get('ma20')
                current_ma60 = trigger_row.get('ma60')
            else:
                # 如果触发日期没有数据，使用最近一天的数据
                latest_row = df_processed.iloc[-1]
                current_price_qfq = latest_row.get('close_qfq')
                current_ma5 = latest_row.get('ma5')
                current_ma10 = latest_row.get('ma10')
                current_ma20 = latest_row.get('ma20')
                current_ma60 = latest_row.get('ma60')
            
            # 识别一字板（需要先计算is_one_word_limit_up）
            limit_up_threshold = self._get_limit_up_threshold(symbol)
            if 'pct_change' not in df_processed.columns or df_processed['pct_change'].isna().all():
                df_processed['pct_change'] = df_processed['close_qfq'].pct_change() * 100
            df_processed['is_limit_up'] = df_processed['pct_change'] >= (limit_up_threshold * 100 - 0.5)
            
            volume_ma_col = 'volume_ma20' if 'volume_ma20' in df_processed.columns else 'v_ma5'
            avg_volume = df_processed['volume'].mean()
            median_volume = df_processed['volume'].median()
            df_processed['is_one_word_limit_up'] = False
            
            for idx in df_processed.index:
                if df_processed.loc[idx, 'is_limit_up']:
                    open_price = df_processed.loc[idx, 'open_qfq']
                    close_price = df_processed.loc[idx, 'close_qfq']
                    high_price = df_processed.loc[idx, 'high_qfq']
                    low_price = df_processed.loc[idx, 'low_qfq']
                    
                    price_diff_ratio = abs(open_price - close_price) / close_price if close_price > 0 else 1.0
                    is_price_match = price_diff_ratio < 0.001
                    
                    if not is_price_match:
                        continue
                    
                    volume = df_processed.loc[idx, 'volume']
                    volume_ma = df_processed.loc[idx, volume_ma_col] if pd.notna(df_processed.loc[idx, volume_ma_col]) else avg_volume
                    volume_ratio = volume / volume_ma if volume_ma > 0 else 1.0
                    
                    # 条件a：完全一字板
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
                        df_processed.loc[idx, 'is_one_word_limit_up'] = True
            
            one_word_stats = self._extract_one_word_stats(df_processed, symbol, trigger_date)
        else:
            one_word_stats = "### 一字板统计信息\n- 出现次数：0次"
        
        # 压缩K线数据为文本摘要（使用处理后的数据，确保技术指标已计算）
        if not df_kline.empty:
            # 使用处理后的数据（包含技术指标）
            kline_data_processed = df_processed.to_dict('records') if 'df_processed' in locals() else kline_data
            kline_summary = self._compress_kline_data(kline_data_processed, symbol, trigger_date, months=kline_months)
        else:
            kline_summary = self._compress_kline_data(kline_data, symbol, trigger_date, months=kline_months)
        
        # 提取模型预测结果（从prediction_result中提取关键信息）
        scores = prediction_result.get('scores', {})
        features = prediction_result.get('features', {})
        
        model_summary = {
            'consecutive_count': features.get('consecutive_count', 0),  # 连板高度
            'pre_position': features.get('pre_position', 'unknown'),  # 位置判断（高位/低位/中位）
            'final_prob': scores.get('final_prob', 0.0),  # 模型预测的二波概率
            'pattern_success_rate': scores.get('pattern_success_rate', 0.0),  # 历史成功率
            'conclusion': prediction_result.get('conclusion', '')  # 模型结论
        }
        
        # 获取股票名称和板块信息（目前为TODO，需要后续完善）
        stock_name = ''  # TODO: 从数据库或其他来源获取股票名称
        sector_name = None  # TODO: 从features或市场情绪服务获取板块名称
        
        return {
            'symbol': symbol,
            'stock_name': stock_name,
            'trigger_date': trigger_date,
            'kline_summary': kline_summary,
            'one_word_stats': one_word_stats,  # 新增：一字板统计信息
            'model_summary': model_summary,
            'use_web_search': use_web_search,
            'sector_name': sector_name,
            'current_price_qfq': current_price_qfq,  # 新增：触发日期当前价格（前复权）
            'current_ma5': current_ma5,  # 新增：当前MA5（前复权）
            'current_ma10': current_ma10,  # 新增：当前MA10（前复权）
            'current_ma20': current_ma20,  # 新增：当前MA20（前复权）
            'current_ma60': current_ma60  # 新增：当前MA60（前复权）
        }
    
    def _format_prompt(self, context: Dict) -> str:
        """
        格式化Prompt，将上下文数据填充到模板中
        
        Args:
            context: 上下文字典（由_prepare_context生成，必须包含所有模板所需的字段）
        
        Returns:
            str: 格式化后的完整Prompt文本
        
        注意：
        - 使用Python的str.format方法填充模板变量
        - 模板文件：stockainews/prompts/stock_rerank_prompt.txt
        - 如果context中缺少字段，会抛出KeyError
        """
        # 构建当前价格信息（前复权），用于在Prompt中明确提供
        current_price_info = ""
        if context.get('current_price_qfq') is not None:
            price_info_parts = [f"当前价格（前复权）：{context['current_price_qfq']:.2f}元"]
            if context.get('current_ma5') is not None:
                price_info_parts.append(f"MA5：{context['current_ma5']:.2f}元")
            if context.get('current_ma10') is not None:
                price_info_parts.append(f"MA10：{context['current_ma10']:.2f}元")
            if context.get('current_ma20') is not None:
                price_info_parts.append(f"MA20：{context['current_ma20']:.2f}元")
            if context.get('current_ma60') is not None:
                price_info_parts.append(f"MA60：{context['current_ma60']:.2f}元")
            current_price_info = "，".join(price_info_parts)
        else:
            current_price_info = "当前价格：无法获取（请从K线数据中查看）"
        
        prompt = self.prompt_template.format(
            symbol=context['symbol'],
            stock_name=context.get('stock_name', ''),
            trigger_date=context['trigger_date'],
            current_price_info=current_price_info,  # 新增：当前价格信息（前复权）
            kline_summary=context['kline_summary'],
            one_word_stats=context.get('one_word_stats', '### 一字板统计信息\n- 出现次数：0次'),  # 新增：一字板统计信息
            consecutive_count=context['model_summary']['consecutive_count'],
            pre_position=context['model_summary']['pre_position'],
            final_prob=context['model_summary']['final_prob'],
            pattern_success_rate=context['model_summary']['pattern_success_rate'],
            conclusion=context['model_summary']['conclusion'],
            stock_news_summary=context['web_search_results'].get('stock_news', '无'),
            market_sentiment_summary=context['web_search_results'].get('market_sentiment', '无'),
            sector_hotspot_summary=context['web_search_results'].get('sector_hotspot', '无')
        )
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """
        解析LLM响应（JSON格式）
        
        尝试多种方式解析LLM返回的JSON：
        1. 直接解析（如果响应以{开头）
        2. 提取JSON（如果响应包含markdown代码块）
        3. 如果都失败，返回默认结构
        
        Args:
            response: LLM返回的响应文本
        
        Returns:
            Dict: 解析后的JSON字典，如果解析失败则返回默认结构
        
        注意：
        - LLM可能返回纯JSON或包含markdown代码块的JSON
        - 解析失败时会记录错误日志（只记录前500字符避免日志过长）
        - 返回默认结构确保程序继续执行，但rerank_score=0.5表示不确定
        """
        try:
            # 尝试直接解析JSON（如果响应以{开头）
            if response.strip().startswith('{'):
                return json.loads(response)
            
            # 尝试提取JSON（去除markdown代码块，如```json ... ```）
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            
            raise ValueError("未找到有效的JSON格式")
        except Exception as e:
            logger.error(f"解析LLM响应失败: {e}, 响应内容: {response[:500]}")  # 只记录前500字符
            # 返回默认结构（确保程序继续执行）
            return {
                'rerank_score': 0.5,
                'risk_level': 'medium',
                'position_judgment': 'middle',
                'is_bottom': False,
                'relay_potential': 'moderate',
                'market_environment': 'neutral',
                'kline_analysis': {  # 新增：K线形态分析
                    'breakthrough_days': [],
                    'top_confirmation_days': [],
                    'is_bottom_confirmed': False,
                    'bottom_reason': '解析失败',
                    'is_high_risk': False,
                    'high_risk_reason': '解析失败'
                },
                'one_word_risk_analysis': {  # 新增：一字板风险分析
                    'has_risk': False,
                    'risk_level': 'low',
                    'analysis': '解析失败',
                    'recommendation': '解析失败'
                },
                'reasons': {
                    'position_risk': '解析失败',
                    'relay_analysis': '解析失败',
                    'bottom_judgment': '解析失败',
                    'market_context': '解析失败'
                },
                'recommendation': 'hold',
                'investment_advice': '解析失败，请检查LLM响应格式',
                'action_summary': '建议观望',
                'confidence': 0.0,
                'key_risks': ['LLM响应解析失败'],
                'key_opportunities': []
            }
    
    def _fix_price_annotation_format(self, rerank_result: Dict, current_price_qfq: Optional[float] = None) -> Dict:
        """
        修正价格标注格式，自动补充缺失的括号
        
        检查并修正以下格式问题：
        1. "XX元前复权" -> "XX元（前复权）"（缺少括号）
        2. "XX元（前复）" -> "XX元（前复权）"（不完整）
        
        Args:
            rerank_result: LLM返回的结果字典
            current_price_qfq: 当前价格（前复权），用于验证价格合理性
        
        Returns:
            Dict: 修正后的结果字典
        """
        import re
        
        # 需要检查的字段
        fields_to_check = [
            'investment_advice',
            'high_risk_reason',
            'bottom_reason',
            'position_risk',
            'relay_analysis',
            'bottom_judgment',
            'market_context'
        ]
        
        # 检查kline_analysis中的字段
        if 'kline_analysis' in rerank_result:
            kline_analysis = rerank_result['kline_analysis']
            if 'high_risk_reason' in kline_analysis:
                fields_to_check.append(('kline_analysis', 'high_risk_reason'))
            if 'bottom_reason' in kline_analysis:
                fields_to_check.append(('kline_analysis', 'bottom_reason'))
            
            # 检查top_confirmation_days中的description
            if 'top_confirmation_days' in kline_analysis:
                for top_day in kline_analysis['top_confirmation_days']:
                    if 'description' in top_day:
                        # 修正格式
                        original = top_day['description']
                        fixed = self._fix_price_text(original)
                        if fixed != original:
                            top_day['description'] = fixed
                            logger.warning(f"修正top_confirmation_days.description中的价格标注格式")
            
            # 检查breakthrough_days中的description
            if 'breakthrough_days' in kline_analysis:
                for breakthrough_day in kline_analysis['breakthrough_days']:
                    if 'description' in breakthrough_day:
                        original = breakthrough_day['description']
                        fixed = self._fix_price_text(original)
                        if fixed != original:
                            breakthrough_day['description'] = fixed
                            logger.warning(f"修正breakthrough_days.description中的价格标注格式")
        
        # 修正各个字段
        for field in fields_to_check:
            if isinstance(field, tuple):
                # 嵌套字段，如('kline_analysis', 'high_risk_reason')
                if field[0] in rerank_result and field[1] in rerank_result[field[0]]:
                    original = rerank_result[field[0]][field[1]]
                    if isinstance(original, str):
                        fixed = self._fix_price_text(original)
                        if fixed != original:
                            rerank_result[field[0]][field[1]] = fixed
                            logger.warning(f"修正{field[0]}.{field[1]}中的价格标注格式")
            else:
                # 直接字段
                if field in rerank_result:
                    original = rerank_result[field]
                    if isinstance(original, str):
                        fixed = self._fix_price_text(original)
                        if fixed != original:
                            rerank_result[field] = fixed
                            logger.warning(f"修正{field}中的价格标注格式")
        
        return rerank_result
    
    def _fix_price_text(self, text: str) -> str:
        """
        修正文本中的价格标注格式
        
        修正规则：
        1. "XX元前复权" -> "XX元（前复权）"
        2. "XX元（前复）" -> "XX元（前复权）"
        
        Args:
            text: 需要修正的文本
        
        Returns:
            str: 修正后的文本
        """
        import re
        
        # 修正"XX元前复权" -> "XX元（前复权）"
        # 匹配模式：数字.数字 元前复权（没有括号）
        pattern1 = r'(\d+\.?\d*)\s*元\s*前复权'
        text = re.sub(pattern1, r'\1元（前复权）', text)
        
        # 修正"XX元（前复）" -> "XX元（前复权）"（不完整）
        pattern2 = r'(\d+\.?\d*)\s*元\s*（前复）'
        text = re.sub(pattern2, r'\1元（前复权）', text)
        
        return text
    
    async def rerank_stock(
        self, 
        prediction_result: Dict,
        use_web_search: bool = True,
        kline_months: int = 6
    ) -> Dict:
        """
        对单个股票进行LLM二次裁判（核心方法）
        
        完整流程：
        1. 准备上下文（K线数据、模型预测结果）
        2. 获取实时市场信息（web_search，可选）
        3. 格式化Prompt并调用LLM
        4. 解析LLM响应
        5. 融合模型预测和LLM评分
        6. 构建增强后的结果
        
        Args:
            prediction_result: predict_stock.py 的输出结果，必须包含symbol、trigger_date、features、scores、conclusion等
            use_web_search: 是否使用web_search获取实时市场信息（默认True，会消耗API配额）
            kline_months: K线数据保留最近N个月（默认6个月，平衡信息量和token消耗）
        
        Returns:
            Dict: 增强后的结果，在原有prediction_result基础上添加：
                - rerank: LLM二次裁判结果（包含rerank_score、risk_level、investment_advice等）
                - final_score_after_rerank: 融合后的最终评分（模型60% + LLM 40%）
                - final_recommendation: 最终投资建议（strong_buy/buy/hold/avoid）
                - final_investment_advice: 最终投资建议文本
        
        异常处理：
        - 如果二次裁判过程出错，会记录错误日志并返回原始结果（添加error标记）
        - 确保即使LLM调用失败，也能返回可用的结果结构
        """
        try:
            # 准备上下文
            context = self._prepare_context(prediction_result, use_web_search=use_web_search, kline_months=kline_months)
            
            # Web搜索（异步）
            if use_web_search:
                web_results = await self._web_search_market_info(
                    context['symbol'],
                    context.get('stock_name', ''),
                    context['trigger_date'],
                    context.get('sector_name')
                )
                context['web_search_results'] = web_results
            else:
                context['web_search_results'] = {
                    'stock_news': '未使用web_search',
                    'market_sentiment': '未使用web_search',
                    'sector_hotspot': '未使用web_search'
                }
            
            # 格式化Prompt
            prompt = self._format_prompt(context)
            
            # 调用LLM
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            
            # 使用JSON格式输出
            chat_result = await self.llm_adapter.generate(
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # 解析响应
            if chat_result.generations and len(chat_result.generations) > 0:
                generation = chat_result.generations[0]
                if isinstance(generation, list) and len(generation) > 0:
                    response_text = generation[0].text
                elif hasattr(generation, 'text'):
                    response_text = generation.text
                elif hasattr(generation, 'message') and hasattr(generation.message, 'content'):
                    response_text = generation.message.content
                else:
                    raise ValueError(f"无法解析LLM响应: {generation}")
            else:
                raise ValueError("LLM响应为空")
            
            rerank_result = self._parse_llm_response(response_text)
            
            # 修正价格标注格式（自动补充缺失的括号）
            rerank_result = self._fix_price_annotation_format(rerank_result, context.get('current_price_qfq'))
            
            # 修正LLM返回结果中的价格字段（确保使用前复权价格）
            # 从context中获取正确的当前价格和均线价格
            current_price_qfq = context.get('current_price_qfq')
            current_ma5 = context.get('current_ma5')
            current_ma10 = context.get('current_ma10')
            current_ma20 = context.get('current_ma20')
            current_ma60 = context.get('current_ma60')
            
            # 修正investment_advice中的价格（如果LLM使用了错误的价格）
            if current_price_qfq is not None and 'investment_advice' in rerank_result:
                investment_advice = rerank_result['investment_advice']
                # 检查investment_advice中是否有明显错误的价格（与当前价格差异过大）
                # 如果发现价格差异超过50%，可能是LLM使用了后复权价格，需要修正
                import re
                price_pattern = r'(\d+\.?\d*)\s*元'
                prices_in_advice = re.findall(price_pattern, investment_advice)
                for price_str in prices_in_advice:
                    try:
                        price_val = float(price_str)
                        # 如果价格与当前价格差异超过50%，可能是后复权价格，需要修正
                        price_diff_ratio = abs(price_val - current_price_qfq) / current_price_qfq
                        if price_diff_ratio > 0.5:
                            # 尝试修正：如果价格差异很大，可能是后复权价格，需要转换为前复权
                            # 但这里我们不知道adjust_factor，所以只能记录警告
                            logger.warning(f"LLM返回的价格 {price_val:.2f}元 与当前价格（前复权）{current_price_qfq:.2f}元 差异较大（{price_diff_ratio*100:.1f}%），可能使用了后复权价格。已通过Prompt明确提供前复权价格，请检查LLM是否正确理解。")
                    except:
                        pass
            
            # 修正kline_analysis中的high_price（确保使用前复权）
            # 注意：high_price应该已经是前复权（从K线数据描述中提取），但这里不做修正
            # 因为LLM从K线数据描述中提取的价格应该已经是前复权
            
            # 融合评分（模型预测60% + LLM二次裁判40%）
            # 权重设计：更信任模型的历史统计，LLM作为风险识别和实时信息补充
            original_prob = prediction_result.get('final_prob', 0.0)
            rerank_score = rerank_result.get('rerank_score', 0.5)
            final_score = (original_prob * 0.6 + rerank_score * 0.4)  # 加权融合
            
            # 确定最终建议（根据融合评分分段）
            if final_score >= 0.7:
                final_recommendation = 'strong_buy'  # 强烈推荐
            elif final_score >= 0.6:
                final_recommendation = 'buy'  # 推荐
            elif final_score >= 0.4:
                final_recommendation = 'hold'  # 持有/观望
            else:
                final_recommendation = 'avoid'  # 回避
            
            # 构建增强后的结果
            enhanced_result = prediction_result.copy()
            enhanced_result['rerank'] = rerank_result
            enhanced_result['final_score_after_rerank'] = final_score
            enhanced_result['final_recommendation'] = final_recommendation
            enhanced_result['final_investment_advice'] = f"综合模型预测({original_prob:.3f})和LLM二次裁判({rerank_score:.3f})，最终评分{final_score:.3f}。{rerank_result.get('investment_advice', '')}"
            
            return enhanced_result
        
        except Exception as e:
            logger.error(f"二次裁判失败: {e}", exc_info=True)
            # 返回原始结果，添加错误标记（确保程序继续执行）
            result = prediction_result.copy()
            result['rerank'] = {
                'error': str(e),
                'rerank_score': 0.5,  # 默认评分（不确定）
                'recommendation': 'hold'  # 默认建议（观望）
            }
            return result
    
    async def rerank_batch(
        self,
        prediction_results: List[Dict],
        topk: int = 10,
        use_web_search: bool = True,
        kline_months: int = 6
    ) -> List[Dict]:
        """
        批量二次裁判（性能优化：只对TOP-K进行详细LLM裁判）
        
        策略：
        - 按模型预测概率排序
        - 只对TOP-K进行详细LLM裁判（消耗API配额）
        - 其他结果快速过滤（不调用LLM，直接使用模型预测概率）
        - 最后按最终评分重新排序
        
        Args:
            prediction_results: 预测结果列表（predict_stock.py的输出）
            topk: 只对TOP-K进行详细裁判（默认10，其他快速过滤以节省成本）
            use_web_search: 是否使用web_search（默认True，获取实时信息但消耗配额）
            kline_months: K线数据保留最近N个月（默认6个月）
        
        Returns:
            List[Dict]: 增强后的结果列表，按final_score_after_rerank降序排列
        
        注意：
        - 性能优化：只对TOP-K调用LLM，其他快速过滤（节省API配额和时间）
        - 单个股票失败不影响其他股票（继续处理）
        - 最终按融合评分重新排序（可能改变原始顺序）
        """
        # 按final_prob排序（模型预测概率降序）
        sorted_results = sorted(
            prediction_results,
            key=lambda x: float(x.get('final_prob', 0.0) or 0.0),
            reverse=True
        )
        
        # 分离TOP-K和其他结果
        topk_results = sorted_results[:topk]
        other_results = sorted_results[topk:]
        
        enhanced_results = []
        
        # 详细裁判TOP-K（调用LLM，消耗API配额）
        for i, result in enumerate(topk_results, 1):
            logger.info(f"[{i}/{len(topk_results)}] 二次裁判: {result.get('symbol')}")
            try:
                enhanced = await self.rerank_stock(result, use_web_search=use_web_search, kline_months=kline_months)
                enhanced_results.append(enhanced)
            except Exception as e:
                logger.error(f"二次裁判失败 {result.get('symbol')}: {e}")
                enhanced_results.append(result)  # 失败时保留原始结果
        
        # 其他结果快速过滤（不调用LLM，只添加默认rerank字段）
        # 使用模型预测概率作为rerank_score，节省API配额
        for result in other_results:
            result_copy = result.copy()
            result_copy['rerank'] = {
                'rerank_score': result.get('final_prob', 0.5),
                'recommendation': 'hold',
                'note': '未进行详细裁判（非TOP-K）'
            }
            enhanced_results.append(result_copy)
        
        # 按最终评分重新排序（融合评分优先，否则使用原始概率）
        enhanced_results.sort(
            key=lambda x: float(x.get('final_score_after_rerank', x.get('final_prob', 0.0) or 0.0)),
            reverse=True
        )
        
        return enhanced_results


async def main_async():
    """
    异步主函数（命令行入口）
    
    支持两种模式：
    1. 批量处理模式（推荐）：从JSON文件读取预测结果，批量进行二次裁判
    2. 单股票测试模式（TODO）：直接指定股票代码和触发日期（暂未实现）
    
    注意：
    - 使用asyncio.run(main_async())作为同步入口
    - 批量处理模式会按final_prob排序，只对TOP-K进行详细裁判
    - 输出文件路径如果未指定，会自动从输入文件名推导（predictions_trigger_ -> predictions_rerank_）
    """
    parser = argparse.ArgumentParser(description='LLM二次裁判服务')
    parser.add_argument('--input', type=str, help='输入JSON文件路径（predict_stock.py的输出）')
    parser.add_argument('--output', type=str, help='输出JSON文件路径')
    parser.add_argument('--symbol', type=str, help='单个股票代码（测试用）')
    parser.add_argument('--trigger-date', type=str, help='触发日期（与--symbol一起使用）')
    parser.add_argument('--topk', type=int, default=20, help='只对TOP-K进行详细裁判（默认20）')
    parser.add_argument('--no-web-search', action='store_true', help='不使用web_search（节省成本）')
    parser.add_argument('--kline-months', type=int, default=6, help='K线数据保留最近N个月（默认6）')
    
    args = parser.parse_args()
    
    service = LLMRerankService()
    
    if args.symbol:
        # 单股票测试模式（暂未实现）
        # TODO: 需要先运行predict_stock.py获取prediction_result，或直接调用run_prediction
        logger.error("单股票测试模式暂未实现，请先使用predict_stock.py获取预测结果，然后使用--input参数")
        return
    
    if args.input:
        # 批量处理模式（推荐）
        if not os.path.exists(args.input):
            logger.error(f"输入文件不存在: {args.input}")
            return
        
        # 加载预测结果JSON文件
        with open(args.input, 'r', encoding='utf-8') as f:
            prediction_results = json.load(f)
        
        logger.info(f"加载了 {len(prediction_results)} 个预测结果")
        
        # 批量二次裁判（只对TOP-K进行详细LLM裁判）
        enhanced_results = await service.rerank_batch(
            prediction_results,
            topk=args.topk,
            use_web_search=not args.no_web_search,  # --no-web-search标志取反
            kline_months=args.kline_months
        )
        
        # 保存结果（如果未指定输出路径，自动从输入文件名推导）
        output_path = args.output or args.input.replace('predictions_trigger_', 'predictions_rerank_')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 使用print确保输出到控制台（解决日志不显示的问题）
        print(f"\n{'='*60}")
        print(f"结果已保存到: {output_path}")
        print(f"共处理 {len(enhanced_results)} 个结果，其中 {args.topk} 个进行了详细裁判")
        print(f"{'='*60}\n")
        
        logger.info(f"结果已保存到: {output_path}")
        logger.info(f"共处理 {len(enhanced_results)} 个结果，其中 {args.topk} 个进行了详细裁判")
    else:
        parser.print_help()


def main():
    """
    主函数（同步入口）
    
    使用asyncio.run运行异步主函数，确保在非异步环境中也能调用。
    """
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
