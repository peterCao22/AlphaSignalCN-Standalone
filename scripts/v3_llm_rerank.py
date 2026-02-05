#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.3 + LLM Rerank 集成脚本

对V3.3筛选器的TOP-K预测结果进行LLM二次分析，生成实盘买卖建议

核心功能：
1. 读取V3.3预测结果（CSV格式）
2. 获取K线数据（最近3个月）
3. 调用LLM分析K线形态、位置风险、接力潜力
4. 生成实盘投资建议（买入/观望/止损点位）
5. 保存增强后的预测结果

使用方法:
    python scripts/v3_llm_rerank.py --date 2025-01-08 --top-k 20
    
    # 不使用web_search（节省LLM成本）
    python scripts/v3_llm_rerank.py --date 2025-01-08 --top-k 10 --no-web-search
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import psycopg2
from typing import Dict, Any, List

# 添加项目根目录到路径
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stockainews.llm_adapters.doubao_adapter import DoubaoAdapter
from stockainews.core.logger import logger
from langchain_core.messages import HumanMessage

# 数据库配置
DB_HOST = '192.168.21.39'
DB_PORT = 5433
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'
DB_NAME = 'stocks_data'


def load_v3_prediction(date_str: str) -> pd.DataFrame:
    """
    加载V3.3预测结果
    
    Args:
        date_str: T3日期 (YYYY-MM-DD)
    
    Returns:
        DataFrame: 预测结果
    """
    # 查找最新的预测文件
    pred_dir = Path('results/v3_predictions')
    pattern = f'v3_prediction_T0_*_T3_{date_str.replace("-", "")}_ensemble_*.csv'
    
    files = list(pred_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到日期{date_str}的预测文件")
    
    # 按时间戳排序，取最新的
    latest_file = sorted(files, key=lambda x: x.stem.split('_')[-1])[-1]
    
    logger.info(f"加载预测文件: {latest_file.name}")
    df = pd.read_csv(latest_file)
    
    return df


def get_kline_data(instrument: str, end_date: str, months: int = 3) -> pd.DataFrame:
    """
    获取K线数据
    
    Args:
        instrument: 股票代码
        end_date: 结束日期 (YYYY-MM-DD)
        months: 往前取几个月的数据
    
    Returns:
        DataFrame: K线数据
    """
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME
    )
    
    query = """
    SELECT date, open, high, low, close, volume, turn
    FROM kline_all
    WHERE instrument = %s 
      AND date <= %s
    ORDER BY date DESC
    LIMIT %s
    """
    
    # 按交易日计算，约60个交易日/月
    limit = months * 60
    
    df = pd.read_sql(query, conn, params=(instrument, end_date, limit))
    conn.close()
    
    # 反转为时间正序
    df = df.iloc[::-1].reset_index(drop=True)
    
    return df


def compress_kline_summary(kline_df: pd.DataFrame, instrument: str) -> str:
    """
    压缩K线数据为文本摘要
    
    Args:
        kline_df: K线数据
        instrument: 股票代码
    
    Returns:
        str: K线摘要文本
    """
    if len(kline_df) == 0:
        return f"{instrument}: 无K线数据"
    
    # 基本统计
    latest = kline_df.iloc[-1]
    oldest = kline_df.iloc[0]
    
    # 计算涨幅
    total_gain = (latest['close'] / oldest['close'] - 1) * 100
    
    # 最近10天数据
    recent_10 = kline_df.tail(10)
    recent_gain = (recent_10.iloc[-1]['close'] / recent_10.iloc[0]['close'] - 1) * 100
    
    # 识别涨停日（简化判断）
    limit_up_days = []
    for i in range(1, len(kline_df)):
        prev_close = kline_df.iloc[i-1]['close']
        curr_close = kline_df.iloc[i]['close']
        gain = (curr_close / prev_close - 1) * 100
        
        # 判断是否涨停（9.5%阈值）
        if gain >= 9.5:
            limit_up_days.append(kline_df.iloc[i]['date'])
    
    # 构建摘要
    summary = f"""
股票: {instrument}
数据范围: {oldest['date']} ~ {latest['date']} (共{len(kline_df)}个交易日)

价格信息:
- 最新收盘: {latest['close']:.2f}元
- 期间涨幅: {total_gain:+.2f}%
- 最近10日涨幅: {recent_gain:+.2f}%

涨停情况:
- 期间涨停次数: {len(limit_up_days)}次
- 最近涨停日: {limit_up_days[-3:] if limit_up_days else '无'}

最近5日K线:
"""
    
    for i in range(max(0, len(kline_df)-5), len(kline_df)):
        row = kline_df.iloc[i]
        day_gain = (row['close'] / row['open'] - 1) * 100 if row['open'] > 0 else 0
        summary += f"  {row['date']}: 开{row['open']:.2f} 高{row['high']:.2f} 低{row['low']:.2f} 收{row['close']:.2f} (日涨{day_gain:+.2f}%) 换手{row.get('turn', 0):.2f}%\n"
    
    return summary


async def llm_analyze_stock(
    instrument: str,
    stock_name: str,
    prediction_score: float,
    kline_summary: str,
    llm_adapter: DoubaoAdapter,
    use_web_search: bool = False
) -> Dict[str, Any]:
    """
    LLM分析股票
    
    Args:
        instrument: 股票代码
        stock_name: 股票名称
        prediction_score: V3.3模型预测分数
        kline_summary: K线摘要
        llm_adapter: LLM适配器
        use_web_search: 是否使用web搜索
    
    Returns:
        dict: LLM分析结果
    """
    prompt = f"""
你是一个专业的A股短线交易分析师。请分析以下股票的K线形态和位置风险，给出实盘建议。

## 股票信息
代码: {instrument}
名称: {stock_name}
V3.3模型预测3天涨幅: {prediction_score:.2f}%

## K线数据
{kline_summary}

## 分析要求
请从以下4个维度进行分析：

1. **位置风险** (position_risk)
   - 评估当前股价位置（底部/中部/顶部）
   - 相对历史高点的距离
   - 评级: low(低风险) / medium(中风险) / high(高风险)

2. **接力潜力** (continuation_potential)
   - 分析涨停后的接力能力
   - 换手率、量能变化
   - 评级: weak(弱) / moderate(中等) / strong(强)

3. **形态判断** (pattern_judgment)
   - K线形态（突破/整理/回调）
   - 均线支撑情况
   - 描述: 简要形态说明

4. **实盘建议** (trading_advice)
   - 操作建议: buy(买入) / watch(观望) / avoid(规避)
   - 建议买入价: 具体价位
   - 止损价: 具体价位
   - 持仓建议: 轻仓/半仓/重仓
   - 理由: 简要说明

请以JSON格式输出：
{{
    "position_risk": "low/medium/high",
    "continuation_potential": "weak/moderate/strong",
    "pattern_judgment": "形态描述",
    "trading_advice": {{
        "action": "buy/watch/avoid",
        "entry_price": 价格数值,
        "stop_loss": 价格数值,
        "position_size": "轻仓/半仓/重仓",
        "reason": "理由说明"
    }},
    "综合评分": 0-100分
}}
"""
    
    try:
        # 构建消息
        messages = [HumanMessage(content=prompt)]
        
        # 调用LLM
        response = await llm_adapter.generate(
            messages=messages,
            temperature=0.1
        )
        
        # 解析LLM响应（兼容不同的响应结构）
        response_text = None
        if response.generations and len(response.generations) > 0:
            generation = response.generations[0]
            if isinstance(generation, list) and len(generation) > 0:
                response_text = generation[0].text
            elif hasattr(generation, 'text'):
                response_text = generation.text
            elif hasattr(generation, 'message') and hasattr(generation.message, 'content'):
                response_text = generation.message.content
            else:
                raise ValueError(f"无法解析LLM响应结构: {type(generation)}")
        else:
            raise ValueError("LLM响应为空")
        
        # 清理可能的markdown标记
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # 解析JSON
        result = json.loads(response_text)
        return result
        
    except Exception as e:
        logger.error(f"LLM分析失败 {instrument}: {e}")
        # 返回默认结果
        return {
            "position_risk": "medium",
            "continuation_potential": "moderate",
            "pattern_judgment": "分析失败",
            "trading_advice": {
                "action": "watch",
                "entry_price": 0,
                "stop_loss": 0,
                "position_size": "轻仓",
                "reason": "LLM分析失败，建议观望"
            },
            "综合评分": 50
        }


async def process_stock(
    row: pd.Series,
    T3_date: str,
    llm_adapter: DoubaoAdapter,
    use_web_search: bool = False
) -> Dict[str, Any]:
    """
    处理单个股票
    
    Args:
        row: 预测结果行
        T3_date: T3日期
        llm_adapter: LLM适配器
        use_web_search: 是否使用web搜索
    
    Returns:
        dict: 增强后的结果
    """
    instrument = row['instrument']
    stock_name = row['stock_name']
    prediction_score = row['prediction_score']
    
    logger.info(f"处理股票: {instrument} {stock_name}")
    
    # 获取K线数据
    try:
        kline_df = get_kline_data(instrument, T3_date, months=3)
        kline_summary = compress_kline_summary(kline_df, instrument)
    except Exception as e:
        logger.error(f"获取K线失败 {instrument}: {e}")
        kline_summary = f"{instrument}: K线数据获取失败"
    
    # LLM分析
    llm_result = await llm_analyze_stock(
        instrument, stock_name, prediction_score,
        kline_summary, llm_adapter, use_web_search
    )
    
    # 合并结果
    result = {
        'instrument': instrument,
        'stock_name': stock_name,
        'v3_prediction': prediction_score,
        'llm_analysis': llm_result,
        'final_recommendation': llm_result.get('trading_advice', {}).get('action', 'watch')
    }
    
    return result


async def main():
    parser = argparse.ArgumentParser(description='V3.3 + LLM Rerank')
    parser.add_argument('--date', required=True, help='T3日期 (YYYY-MM-DD)')
    parser.add_argument('--top-k', type=int, default=20, help='分析TOP-K个预测结果')
    parser.add_argument('--no-web-search', action='store_true', help='不使用web搜索')
    
    args = parser.parse_args()
    
    logger.info(f"V3.3 + LLM Rerank - T3日期: {args.date}, TOP-K: {args.top_k}")
    
    # 加载V3.3预测结果
    try:
        pred_df = load_v3_prediction(args.date)
        logger.info(f"加载预测结果: {len(pred_df)}只股票")
    except Exception as e:
        logger.error(f"加载预测失败: {e}")
        return
    
    # 取TOP-K
    pred_df = pred_df.head(args.top_k)
    logger.info(f"选取TOP-{args.top_k}进行LLM分析")
    
    # 初始化LLM
    llm_adapter = DoubaoAdapter()
    
    # 处理每个股票
    results = []
    for idx, row in pred_df.iterrows():
        try:
            result = await process_stock(
                row, args.date, llm_adapter,
                use_web_search=not args.no_web_search
            )
            results.append(result)
        except Exception as e:
            logger.error(f"处理股票失败 {row['instrument']}: {e}")
    
    # 保存结果
    output_file = f'results/v3_predictions/v3_llm_rerank_{args.date.replace("-", "")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"保存结果: {output_file}")
    
    # 打印汇总
    logger.info(f"\n{'='*80}")
    logger.info(f"LLM Rerank 完成")
    logger.info(f"{'='*80}")
    logger.info(f"分析股票数: {len(results)}")
    
    buy_count = sum(1 for r in results if r['final_recommendation'] == 'buy')
    watch_count = sum(1 for r in results if r['final_recommendation'] == 'watch')
    avoid_count = sum(1 for r in results if r['final_recommendation'] == 'avoid')
    
    logger.info(f"建议买入: {buy_count}")
    logger.info(f"建议观望: {watch_count}")
    logger.info(f"建议规避: {avoid_count}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
