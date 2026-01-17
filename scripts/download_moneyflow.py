"""
下载资金流向数据（精选15个关键字段 - 优化版）

用于二波预测的核心资金流指标：
- 净流入金额（主力、超大单、大单、小单）- 已预计算 ✓
- 净主动买入额（主力、超大单）- 已预计算 ✓
- 资金流入/流出占比 - 已预计算 ✓
- 原始流入/流出金额（用于衍生计算）

优势：使用 BigQuant 预计算字段，无需手动计算净值，更准确高效

作者：AI Assistant
日期：2026-01-14
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta

# 统一以本仓库根目录为准（AlphaSignalCN-Standalone）
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 导入 BigQuant SDK
try:
    from bigquantdai import dai
except ImportError:
    try:
        from bigquant.api import dai
    except ImportError:
        import dai

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# 精选的15个关键字段（优化版 - 使用预计算字段）
KEY_MONEYFLOW_FIELDS = [
    # 基础字段
    'date',
    'instrument',
    
    # 【最核心】净流入金额（已预计算，直接可用）
    'netflow_amount_main',              # 主力净流入金额
    'netflow_amount_large',             # 超大单净流入金额
    'netflow_amount_big',               # 大单净流入金额
    'netflow_amount_small',             # 小单净流入金额（散户）
    
    # 【次核心】净主动买入（另一种算法）
    'net_active_buy_amount_main',       # 主力净主动买入额
    'net_active_buy_amount_large',      # 超大单净主动买入额
    
    # 【重要】资金流入占比（已预计算）
    'netflow_amount_rate_main',         # 主力净流入占比
    'inflow_amount_rate_main',          # 主力流入占比
    'outflow_amount_rate_main',         # 主力流出占比
    
    # 【辅助】绝对值（用于计算其他指标）
    'inflow_amount_main',               # 主力流入金额
    'outflow_amount_main',              # 主力流出金额
    
    # 【基准】全单汇总
    'active_buy_amount_all',            # 全部主动买入额（作为计算基准）
]


def download_moneyflow_batch(start_date, end_date):
    """
    下载单个批次的资金流向数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame 或 None
    """
    fields_str = ', '.join(KEY_MONEYFLOW_FIELDS)
    sql = f"""
    SELECT {fields_str}
    FROM cn_stock_moneyflow
    """
    
    try:
        logging.info(f"  下载批次: {start_date} ~ {end_date}")
        df = dai.query(sql, filters={"date": [start_date, end_date]}).df()
        
        if df is None or df.empty:
            logging.warning(f"  该批次无数据")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        logging.info(f"  ✓ 获取 {len(df):,} 条数据")
        return df
        
    except Exception as e:
        logging.error(f"  ❌ 该批次下载失败: {e}")
        return None


def download_moneyflow(days_range=365, output_dir='data/raw', batch_days=60):
    """
    下载资金流向数据（分批下载以避免 200MB 限制）
    
    Args:
        days_range: 下载天数范围（默认365天）
        output_dir: 输出目录
        batch_days: 每批下载天数（默认60天，避免超过200MB限制）
    """
    
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(out_dir / "moneyflow.csv")
    
    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_range)
    
    logging.info("=" * 80)
    logging.info("开始下载资金流向数据（分批下载）")
    logging.info("=" * 80)
    logging.info(f"字段数: {len(KEY_MONEYFLOW_FIELDS)} 个（精选关键指标）")
    logging.info(f"日期范围: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({days_range} 天)")
    logging.info(f"批次大小: {batch_days} 天/批（避免 200MB 限制）")
    logging.info(f"输出文件: {output_file}")
    
    # 检查是否已有数据（增量下载）
    existing_df = None
    latest_date = None
    
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            latest_date = existing_df['date'].max()
            logging.info(f"\n✓ 发现已有数据: {len(existing_df)} 条")
            logging.info(f"最新日期: {latest_date.strftime('%Y-%m-%d')}")
            
            # 调整开始日期（只下载新数据）
            if latest_date and latest_date >= start_date:
                start_date = latest_date + timedelta(days=1)
                logging.info(f"调整为增量下载: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
                
                if start_date >= end_date:
                    logging.info("\n✓ 数据已是最新，无需下载")
                    return existing_df
        except Exception as e:
            logging.warning(f"读取已有数据失败: {e}，将全量下载")
            existing_df = None
    
    # 分批下载
    all_batches = []
    current_start = start_date
    batch_num = 0
    
    logging.info(f"\n开始分批下载...")
    
    while current_start < end_date:
        batch_num += 1
        current_end = min(current_start + timedelta(days=batch_days), end_date)
        
        logging.info(f"\n【批次 {batch_num}】")
        batch_df = download_moneyflow_batch(
            current_start.strftime('%Y-%m-%d'),
            current_end.strftime('%Y-%m-%d')
        )
        
        if batch_df is not None:
            all_batches.append(batch_df)
        
        current_start = current_end + timedelta(days=1)
    
    if not all_batches:
        logging.warning("\n未获取到任何数据")
        return existing_df if existing_df is not None else pd.DataFrame()
    
    # 合并所有批次
    logging.info(f"\n合并 {len(all_batches)} 个批次...")
    df = pd.concat(all_batches, ignore_index=True)
    
    logging.info(f"\n✓ 下载成功: {len(df)} 条新数据")
    logging.info(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
    logging.info(f"股票数量: {df['instrument'].nunique()} 只")
    
    # 数据统计
    logging.info(f"\n数据统计:")
    logging.info(f"  - 总记录数: {len(df):,}")
    logging.info(f"  - 字段数: {len(df.columns)}")
    logging.info(f"  - 数据大小: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # 合并新旧数据
    if existing_df is not None:
        logging.info(f"\n合并已有数据...")
        df = pd.concat([existing_df, df], ignore_index=True)
        logging.info(f"去重前: {len(df)} 条")
        
        # 去重（保留最新的记录）
        df = df.drop_duplicates(subset=['date', 'instrument'], keep='last')
        logging.info(f"去重后: {len(df)} 条")
    
    # 按日期排序
    df = df.sort_values(['date', 'instrument']).reset_index(drop=True)
    
    # 保存数据
    df.to_csv(output_file, index=False)
    logging.info(f"\n✓ 已保存: {output_file}")
    
    # 字段有效性统计
    logging.info(f"\n字段有效性:")
    for col in KEY_MONEYFLOW_FIELDS[2:]:  # 跳过 date 和 instrument
        valid_pct = (df[col].notna() & (df[col] != 0)).sum() / len(df) * 100
        logging.info(f"  - {col:35s}: {valid_pct:5.1f}% 有效")
    
    # 数据质量检查
    logging.info(f"\n💡 核心指标统计:")
    logging.info(f"  - netflow_amount_main（主力净流入）: 已预计算 ✓")
    logging.info(f"  - netflow_amount_rate_main（主力净流入占比）: 已预计算 ✓")
    logging.info(f"  - inflow/outflow_amount_rate（流入/流出占比）: 已预计算 ✓")
    
    # 示例统计
    positive_count = (df['netflow_amount_main'] > 0).sum()
    logging.info(f"\n资金流向统计:")
    logging.info(f"  - 主力净流入样本: {positive_count:,} ({positive_count/len(df)*100:.1f}%)")
    logging.info(f"  - 主力净流出样本: {len(df)-positive_count:,} ({(len(df)-positive_count)/len(df)*100:.1f}%)")
    
    # 资金流向强度分布
    strong_inflow = (df['netflow_amount_main'] > df['netflow_amount_main'].quantile(0.75)).sum()
    strong_outflow = (df['netflow_amount_main'] < df['netflow_amount_main'].quantile(0.25)).sum()
    logging.info(f"\n资金流向强度:")
    logging.info(f"  - 强流入（前25%）: {strong_inflow:,} ({strong_inflow/len(df)*100:.1f}%)")
    logging.info(f"  - 强流出（后25%）: {strong_outflow:,} ({strong_outflow/len(df)*100:.1f}%)")
    
    logging.info(f"\n" + "=" * 80)
    logging.info(f"✓ 资金流数据下载完成")
    logging.info(f"=" * 80)
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='下载资金流向数据')
    parser.add_argument('--days', type=int, default=365, help='下载天数范围（默认365=1年）')
    parser.add_argument('--output', type=str, default='data/raw', help='输出目录')
    
    args = parser.parse_args()
    
    download_moneyflow(days_range=args.days, output_dir=args.output)
