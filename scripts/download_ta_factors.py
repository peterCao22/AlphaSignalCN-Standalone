"""
下载 BigQuant 技术分析因子 (cn_stock_factors_ta)

包含移动平均线、RSI、MACD等常用技术指标
"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import logging

# 添加项目根目录到路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_query(dai, sql, max_retries=3):
    """带重试的安全查询"""
    for i in range(max_retries):
        try:
            result = dai.query(sql, full_db_scan=True).df()
            return result
        except Exception as e:
            if i < max_retries - 1:
                logger.warning(f"查询失败，重试 {i+1}/{max_retries}: {e}")
                import time
                time.sleep(2)
            else:
                raise

def query_all_fields():
    """查询 cn_stock_factors_ta 表的所有字段"""
    logger.info("=" * 60)
    logger.info("查询 cn_stock_factors_ta 表的所有字段")
    logger.info("=" * 60)
    
    # 导入 BigQuant SDK
    try:
        from bigquantdai import dai
        logger.info("✓ SDK 导入成功")
    except ImportError:
        try:
            from bigquant.api import dai
            logger.info("✓ SDK 导入成功 (bigquant.api)")
        except ImportError:
            import dai
            logger.info("✓ SDK 导入成功 (dai)")
    
    # 查询表的最新一条数据，获取所有字段名
    sql = """
    SELECT * FROM cn_stock_factors_ta 
    WHERE date = (SELECT MAX(date) FROM cn_stock_factors_ta)
    LIMIT 1
    """
    
    try:
        logger.info("正在查询表结构...")
        df = safe_query(dai, sql)
        
        if df is None or df.empty:
            logger.error("未获取到数据")
            return None
        
        all_fields = df.columns.tolist()
        
        logger.info(f"\n✓ 表共有 {len(all_fields)} 个字段\n")
        
        # 按类别分组显示
        base_fields = [f for f in all_fields if f in ['date', 'instrument']]
        sma_fields = [f for f in all_fields if f.startswith('sma_')]
        ema_fields = [f for f in all_fields if f.startswith('ema_')]
        wma_fields = [f for f in all_fields if f.startswith('wma_')]
        macd_fields = [f for f in all_fields if 'macd' in f]
        kdj_fields = [f for f in all_fields if f.startswith('kdj_')]
        bias_fields = [f for f in all_fields if f.startswith('bias_')]
        rsi_fields = [f for f in all_fields if f.startswith('rsi')]
        bbands_fields = [f for f in all_fields if 'bbands' in f or 'boll' in f]
        cci_fields = [f for f in all_fields if f.startswith('cci')]
        willr_fields = [f for f in all_fields if 'willr' in f or f.startswith('wr_')]
        other_fields = [f for f in all_fields if f not in base_fields + sma_fields + ema_fields + wma_fields + 
                       macd_fields + kdj_fields + bias_fields + rsi_fields + bbands_fields + cci_fields + willr_fields]
        
        # 打印分类结果
        logger.info(f"【基础字段】({len(base_fields)} 个):")
        for f in base_fields:
            logger.info(f"  - {f}")
        
        logger.info(f"\n【SMA 简单移动平均线】({len(sma_fields)} 个):")
        for f in sorted(sma_fields):
            logger.info(f"  - {f}")
        
        logger.info(f"\n【EMA 指数移动平均线】({len(ema_fields)} 个):")
        for f in sorted(ema_fields):
            logger.info(f"  - {f}")
        
        logger.info(f"\n【WMA 加权移动平均线】({len(wma_fields)} 个):")
        for f in sorted(wma_fields):
            logger.info(f"  - {f}")
        
        logger.info(f"\n【MACD 指标】({len(macd_fields)} 个):")
        for f in sorted(macd_fields):
            logger.info(f"  - {f}")
        
        logger.info(f"\n【KDJ 指标】({len(kdj_fields)} 个):")
        for f in sorted(kdj_fields):
            logger.info(f"  - {f}")
        
        logger.info(f"\n【BIAS 乖离率】({len(bias_fields)} 个):")
        for f in sorted(bias_fields):
            logger.info(f"  - {f}")
        
        logger.info(f"\n【RSI 相对强弱指标】({len(rsi_fields)} 个):")
        for f in sorted(rsi_fields):
            logger.info(f"  - {f}")
        
        logger.info(f"\n【布林带/BBANDS】({len(bbands_fields)} 个):")
        for f in sorted(bbands_fields):
            logger.info(f"  - {f}")
        
        logger.info(f"\n【CCI 指标】({len(cci_fields)} 个):")
        for f in sorted(cci_fields):
            logger.info(f"  - {f}")
        
        logger.info(f"\n【威廉指标/WILLR】({len(willr_fields)} 个):")
        for f in sorted(willr_fields):
            logger.info(f"  - {f}")
        
        if other_fields:
            logger.info(f"\n【其他指标】({len(other_fields)} 个):")
            for f in sorted(other_fields):
                logger.info(f"  - {f}")
        
        return all_fields
        
    except Exception as e:
        logger.error(f"查询失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def download_ta_factors(days=90):
    """
    下载技术分析因子数据
    
    Args:
        days: 下载最近多少天的数据（默认90天）
    """
    logger.info("=" * 60)
    logger.info("开始下载技术分析因子数据")
    logger.info("=" * 60)
    
    # 导入 BigQuant SDK
    try:
        from bigquantdai import dai
        logger.info("✓ 使用 bigquantdai SDK")
    except ImportError:
        try:
            from bigquant.api import dai
            logger.info("✓ 使用 bigquant.api SDK")
        except ImportError:
            import dai
            logger.info("✓ 使用 dai 模块")
    
    # 计算日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 输出目录
    output_dir = os.path.join(base_dir, 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'ta_factors.csv')
    
    # 检查现有数据，实现增量下载
    existing_df = None
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            if not existing_df.empty and 'date' in existing_df.columns:
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                latest_date = existing_df['date'].max()
                logger.info(f"发现已有数据，最新日期: {latest_date.strftime('%Y-%m-%d')}")
                
                # 只下载新数据
                download_start = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
                # SQL 使用 date >= start_date AND date <= end_date（两端都包含）
                # 因此 download_start == end_date 时仍需要下载当天数据
                if download_start > end_date:
                    logger.info(f"✓ 数据已是最新，无需下载")
                    return
                
                start_date = download_start
                logger.info(f"执行增量下载: {start_date} 至 {end_date}")
        except Exception as e:
            logger.warning(f"读取已有数据失败: {e}，将进行全量下载")
    
    # 下载数据
    logger.info(f"下载技术分析因子: {start_date} 至 {end_date}")
    
    # 【修正】表中实际有113个字段，包含所有常用技术指标！
    # 选择最有价值的字段组合：
    
    sql = f"""
    SELECT 
        instrument, date,
        -- 简单移动平均线（SMA）6个
        sma_5, sma_10, sma_20, sma_60, sma_120, sma_250,
        -- 指数移动平均线（EMA）6个
        ema_5, ema_10, ema_20, ema_60, ema_120, ema_250,
        -- 加权移动平均线（WMA）6个
        wma_5, wma_10, wma_20, wma_60, wma_120, wma_250,
        -- MACD（标准12-26-9参数）3个
        macd_diff_12_26_9, macd_dea_12_26_9, macd_hist_12_26_9,
        -- MACD（快速5-20-5参数）3个
        macd_diff_5_20_5, macd_dea_5_20_5, macd_hist_5_20_5,
        -- 乖离率（BIAS）6个
        bias_5, bias_10, bias_20, bias_60, bias_120, bias_250,
        -- KDJ（标准9-3-3参数）3个
        kdj_k_9_3_3, kdj_d_9_3_3, kdj_j_9_3_3,
        -- KDJ（快速5-3-3参数）3个
        kdj_k_5_3_3, kdj_d_5_3_3, kdj_j_5_3_3,
        -- RSI（多周期）6个
        rsi_6, rsi_12, rsi_24, rsi_48, rsi_60, rsi_120,
        -- 布林带（标准20日，2倍标准差）3个
        bbands_upper_20_2, bbands_middle_20_2, bbands_lower_20_2,
        -- 布林带（10日，2倍标准差）3个
        bbands_upper_10_2, bbands_middle_10_2, bbands_lower_10_2,
        -- CCI 指标 3个
        cci_14, cci_20, cci_60,
        -- ATR 真实波动幅度 3个
        atr_14, atr_20, atr_60,
        -- OBV 能量潮 2个
        obv, maobv_20,
        -- TRIX 三重指数平滑 2个
        trix_12_9, trix_20_12
    FROM cn_stock_factors_ta
    WHERE date >= '{start_date}' AND date <= '{end_date}'
    """
    
    logger.info("✓ 下载字段: SMA/EMA/WMA、MACD、KDJ、RSI、布林带、CCI、ATR、OBV、TRIX")
    
    try:
        df = safe_query(dai, sql)
        
        if df is None or df.empty:
            logger.warning(f"未获取到数据")
            return
        
        logger.info(f"✓ 下载完成: {len(df)} 条记录")
        logger.info(f"  - 日期范围: {df['date'].min()} 至 {df['date'].max()}")
        logger.info(f"  - 股票数量: {df['instrument'].nunique()}")
        logger.info(f"  - 字段数量: {len(df.columns)}")
        
        # 【计算 KDJ 的 J 值】如果表中没有提供
        # J = 3*K - 2*D
        if 'kdj_k_9_3_3' in df.columns and 'kdj_d_9_3_3' in df.columns and 'kdj_j_9_3_3' not in df.columns:
            df['kdj_j_9_3_3'] = 3 * df['kdj_k_9_3_3'] - 2 * df['kdj_d_9_3_3']
            logger.info("✓ 已计算 KDJ 的 J 值")
        
        # 显示所有字段名
        logger.info("\n包含的技术指标字段:")
        tech_cols = [col for col in df.columns if col not in ['instrument', 'date']]
        for i, col in enumerate(sorted(tech_cols), 1):
            logger.info(f"  {i:2d}. {col}")
        
        # 合并新旧数据
        if existing_df is not None and not existing_df.empty:
            logger.info("合并新旧数据...")
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['instrument', 'date'], keep='last')
            df = combined_df
            logger.info(f"合并后: {len(df)} 条记录")
        
        # 数据保留策略：只保留最近180天的数据
        df['date'] = pd.to_datetime(df['date'])
        cutoff_date = datetime.now() - timedelta(days=180)
        df = df[df['date'] >= cutoff_date]
        logger.info(f"应用保留策略（180天）: {len(df)} 条记录")
        
        # 保存
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"✓ 已保存: {output_file}")
        
        # 显示字段统计
        logger.info("\n字段统计:")
        for col in df.columns:
            if col not in ['instrument', 'date']:
                non_null_count = df[col].notna().sum()
                non_null_pct = non_null_count / len(df) * 100
                logger.info(f"  - {col}: {non_null_pct:.1f}% 有效")
        
    except Exception as e:
        logger.error(f"下载失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    import sys
    
    # 如果传入 --list-fields 参数，只查询所有字段
    if len(sys.argv) > 1 and sys.argv[1] == '--list-fields':
        fields = query_all_fields()
        if fields:
            logger.info("\n" + "=" * 60)
            logger.info(f"✓ 共找到 {len(fields)} 个字段")
            logger.info("=" * 60)
    else:
        download_ta_factors(days=90)
        logger.info("\n" + "=" * 60)
        logger.info("下载完成")
        logger.info("=" * 60)

if __name__ == "__main__":
    main()
