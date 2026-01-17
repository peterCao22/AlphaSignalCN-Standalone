import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta, date as date_type
import sys
import argparse
import asyncio

from pathlib import Path

# =========================
# 项目根目录统一口径
# - 以本仓库（AlphaSignalCN-Standalone）为根目录
# - 避免硬编码到其他仓库路径
# =========================
REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = REPO_ROOT / "results"
TEMP_DIR = DATA_DIR / "temp"

# 确保路径在导入前添加（pattern_matcher.py 位于 scripts/）
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

# 导入本地模块
from pattern_matcher import PatternMatcher  # type: ignore

# 导入数据补充服务
from stockainews.adapters.moma_adapter import MomaAdapter
from stockainews.services.layer3_data_supplement import Layer3DataSupplement

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_latest_file(directory, pattern='*.csv'):
    """
    获取目录中最新修改的文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式（如'*.csv'或'kline*.csv'）
    
    Returns:
        最新文件的完整路径，如果没有找到则返回None
    """
    from pathlib import Path
    import glob
    
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    
    # 查找匹配的文件
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)

def get_kline_main_file(kline_dir: str) -> str:
    """
    获取K线主文件路径（优先使用 kline_all.csv）。
    - 约定：预测/补数检查默认使用 kline_all.csv，避免误读 kline_*.csv 或其他派生文件。
    - 兼容：若 kline_all.csv 不存在，则回退到目录里最新的 kline*.csv（并打印警告）。
    """
    from pathlib import Path
    kdir = Path(kline_dir)
    main_path = kdir / "kline_all.csv"
    if main_path.exists():
        return str(main_path)

    fallback = get_latest_file(kline_dir, "kline*.csv")
    if fallback:
        logging.warning(f"未找到 kline_all.csv，回退使用最新K线文件: {fallback}")
        return fallback

    return ""


def fetch_local_data(symbol):
    logging.info(f"正在从本地加载 {symbol} 的数据...")
    
    # 1. 【优化】加载 K 线数据：默认只读 kline_all.csv（避免误读其他文件）
    kline_dir = str(RAW_DIR / "kline")
    kline_path = get_kline_main_file(kline_dir)
    
    df_kline = None
    if kline_path:
        try:
            logging.info(f"正在读取K线数据: {kline_path}")
            df_kline_all = pd.read_csv(kline_path, dtype={'stock_code': str})
            df_kline = df_kline_all[df_kline_all['instrument'] == symbol].copy()
            if not df_kline.empty:
                # 获取文件修改时间
                from pathlib import Path
                import datetime as dt
                mod_time = dt.datetime.fromtimestamp(Path(kline_path).stat().st_mtime)
                logging.info(f"成功从 {kline_path} 加载K线数据: {len(df_kline)} 条记录（文件更新时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}）")
            else:
                logging.warning(f"{kline_path} 中未找到 {symbol} 的数据")
        except Exception as e:
            logging.error(f"读取 {kline_path} 失败: {e}")
    
    if df_kline is None or df_kline.empty:
        logging.error(f"未找到 {symbol} 的K线数据")
        return None, None, None
    
    # 2. 【优化】加载筹码数据：智能选择最新文件
    chips_dir = str(RAW_DIR)
    chips_path = get_latest_file(chips_dir, 'chips*.csv')
    
    df_chips = pd.DataFrame()
    if chips_path:
        try:
            df_chips_all = pd.read_csv(chips_path)
            # 【修复】使用 format='ISO8601' 处理混合日期格式（兼容 2026-1-5 0:00 和 2021-1-4）
            if 'date' in df_chips_all.columns:
                df_chips_all['date'] = pd.to_datetime(df_chips_all['date'], format='ISO8601')
            df_chips = df_chips_all[df_chips_all['instrument'] == symbol].copy()
            if not df_chips.empty:
                logging.info(f"成功加载筹码数据: {len(df_chips)} 条记录（文件: {os.path.basename(chips_path)}）")
        except Exception as e:
            logging.warning(f"读取筹码数据失败: {e}")
    else:
        logging.warning("未找到筹码数据文件")
    
    # 3. 【优化】加载龙虎榜数据：智能选择最新文件
    dragon_dir = str(RAW_DIR / "limit_up")
    dragon_path = get_latest_file(dragon_dir, 'dragon_list*.csv')
    
    df_dragon = pd.DataFrame()
    if dragon_path:
        try:
            df_dragon_all = pd.read_csv(dragon_path)
            df_dragon = df_dragon_all[df_dragon_all['instrument'] == symbol].copy()
            if not df_dragon.empty:
                logging.info(f"成功加载龙虎榜数据: {len(df_dragon)} 条记录（文件: {os.path.basename(dragon_path)}）")
        except Exception as e:
            logging.warning(f"读取龙虎榜数据失败: {e}")
    else:
        logging.warning("未找到龙虎榜数据文件")
    
    return df_kline, df_chips, df_dragon

def process_kline(df):
    logging.info("计算技术指标...")
    # 使用 format='ISO8601' 处理不同的日期格式（支持 YYYY-MM-DD 和 YYYY-MM-DD HH:MM:SS）
    df['date'] = pd.to_datetime(df['date'], format='ISO8601')
    df = df.sort_values('date').reset_index(drop=True)
    
    # 计算前复权价格
    # 如果数据已经是前复权的（来自补充数据），直接使用原价格
    if 'adjust_factor' in df.columns:
        latest_factor = df['adjust_factor'].iloc[-1]
        for col in ['open', 'high', 'low', 'close']:
            df[f'{col}_qfq'] = df[col] / latest_factor
    else:
        # 数据已经是前复权的，直接复制
        for col in ['open', 'high', 'low', 'close']:
            df[f'{col}_qfq'] = df[col]

    # === 处理停牌/无交易日：只在“有效交易日”上计算技术指标 ===
    # BigQuant 停牌日常见表现：close/open/high/low 为 NaN，volume=0，amount=NaN
    # 若直接 rolling，会导致后续多个交易日的 MA/量比/RSI 全部 NaN。
    df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce')
    df['amount'] = pd.to_numeric(df.get('amount', np.nan), errors='coerce')
    df['close_qfq'] = pd.to_numeric(df.get('close_qfq', np.nan), errors='coerce')

    valid_mask = df['close_qfq'].notna() & (df['volume'].fillna(0) > 0)

    # 先初始化指标列（无效日保持 NaN，不参与触发判断）
    for col in ['ma5', 'ma10', 'ma20', 'ma60', 'v_ma5', 'volume_ratio', 'pct_change', 'rsi']:
        df[col] = np.nan

    df_valid = df.loc[valid_mask].copy()
    if not df_valid.empty:
        # 均线（按“有效交易日序列”滚动）
        df_valid['ma5'] = df_valid['close_qfq'].rolling(window=5).mean()
        df_valid['ma10'] = df_valid['close_qfq'].rolling(window=10).mean()
        df_valid['ma20'] = df_valid['close_qfq'].rolling(window=20).mean()
        df_valid['ma60'] = df_valid['close_qfq'].rolling(window=60).mean()

        # 成交量均线/量比（避免分母为0 -> inf）
        df_valid['v_ma5'] = df_valid['volume'].rolling(window=5).mean()
        denom = df_valid['v_ma5'].shift(1)
        df_valid['volume_ratio'] = np.where(denom > 0, df_valid['volume'] / denom, np.nan)

        # 涨跌幅（关闭 fill_method 以消除 FutureWarning）
        df_valid['pct_change'] = df_valid['close_qfq'].pct_change(fill_method=None) * 100

        # RSI（有效交易日序列）
        delta = df_valid['close_qfq'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df_valid['rsi'] = 100 - (100 / (1 + rs))

        # 回写到原 df（按 index 对齐）
        for col in ['ma5', 'ma10', 'ma20', 'ma60', 'v_ma5', 'volume_ratio', 'pct_change', 'rsi']:
            df.loc[df_valid.index, col] = df_valid[col]
    
    # 标记涨停（根据板块不同设置不同阈值）
    instrument = df['instrument'].iloc[0]
    stock_code = instrument.split('.')[0] if '.' in instrument else instrument
    
    # 判断涨停阈值
    if stock_code.startswith(('92', '43', '8')):
        # 北交所：30%涨停
        limit = 29.5
    elif stock_code.startswith(('300', '301', '688', '689')):
        # 创业板、科创板：20%涨停
        limit = 19.5
    else:
        # 主板（沪深）：10%涨停
        limit = 9.5
    
    df['is_limit_up'] = (df['pct_change'] >= limit) & df['pct_change'].notna()
    
    return df

def get_latest_trading_date():
    """
    获取最近一个交易日（不包括今天）
    
    Returns:
        datetime: 最近一个交易日
    """
    now = datetime.now()
    # 如果今天是交易日且已经16:00后，最近交易日是今天
    if is_trading_day(now) and now.hour >= 16:
        return now
    # 否则往前找最近的交易日
    prev_td = prev_trading_day(now)
    return datetime.combine(prev_td, datetime.min.time())


# === 交易日历（BigQuant all_trading_days）===
_TRADING_DAYS_CACHE: dict[int, set[date_type]] = {}
_TRADING_DAYS_CACHE_FAILED: set[int] = set()


def _load_trading_days_from_bigquant(year: int) -> set[date_type] | None:
    """
    从 BigQuant 的 all_trading_days 表加载指定年份交易日集合（date 对象）。
    失败则返回 None（调用方需回退到 weekday 规则）。
    """
    if year in _TRADING_DAYS_CACHE:
        return _TRADING_DAYS_CACHE[year]
    if year in _TRADING_DAYS_CACHE_FAILED:
        return None

    try:
        # 延迟导入，避免无配额/无依赖时影响主流程
        from bigquantdai import dai

        start = f"{year}-01-01"
        end = f"{year}-12-31"
        sql = f"""
        SELECT date
        FROM all_trading_days
        WHERE date >= '{start}' AND date <= '{end}'
        """
        df = dai.query(sql, full_db_scan=True).df()
        if df is None or df.empty:
            logging.warning(f"BigQuant all_trading_days 返回为空（year={year}），将回退到 weekday 规则")
            _TRADING_DAYS_CACHE_FAILED.add(year)
            return None

        col = 'date' if 'date' in df.columns else df.columns[0]
        ds = pd.to_datetime(df[col], errors='coerce').dt.date
        days = set([d for d in ds.tolist() if d is not None])
        if not days:
            logging.warning(f"BigQuant all_trading_days 解析后为空（year={year}），将回退到 weekday 规则")
            _TRADING_DAYS_CACHE_FAILED.add(year)
            return None

        _TRADING_DAYS_CACHE[year] = days
        logging.info(f"[OK] 已加载 BigQuant 交易日历：{year} 年共 {len(days)} 天")
        return days
    except Exception as e:
        logging.warning(f"加载 BigQuant 交易日历失败（year={year}）：{e}，将回退到 weekday 规则")
        _TRADING_DAYS_CACHE_FAILED.add(year)
        return None


def is_trading_day(d: datetime | date_type) -> bool:
    """
    判断是否交易日：优先 BigQuant all_trading_days；失败则回退 weekday<5。
    """
    dd = d.date() if isinstance(d, datetime) else d
    days = _load_trading_days_from_bigquant(dd.year)
    if days is not None:
        return dd in days
    return dd.weekday() < 5


def prev_trading_day(d: datetime | date_type) -> date_type:
    """
    获取 d 之前（不含 d 当天）的最近一个交易日。
    """
    dd = d.date() if isinstance(d, datetime) else d
    cur = dd - timedelta(days=1)
    # 最多回溯 370 天防死循环
    for _ in range(370):
        if is_trading_day(cur):
            return cur
        cur = cur - timedelta(days=1)
    # 极端兜底：回退 weekday 规则
    while cur.weekday() >= 5:
        cur = cur - timedelta(days=1)
    return cur


def _pick_limit_up_target_date(now: datetime, force_today: bool) -> datetime:
    """
    统一封装“涨停池目标日期选择”逻辑：交易日判定使用 BigQuant 日历（可回退）。
    """
    is_td = is_trading_day(now)
    is_data_ready = now.hour >= 16  # 16:00后数据才稳定可用

    if force_today and is_td:
        return now
    if is_td and is_data_ready:
        return now
    # 否则取最近一个交易日（不含今天）
    return datetime.combine(prev_trading_day(now), datetime.min.time())


async def check_and_supplement_data(symbol, supplement_service):
    """
    检查数据是否完整，如果缺失则补充
    
    Args:
        symbol: 股票代码（如"300034.SZ"）
        supplement_service: Layer3DataSupplement实例
    
    Returns:
        bool: 是否需要重新加载数据
    """
    stock_code = symbol.split('.')[0]  # 提取6位股票代码
    
    # 【优化】检查K线数据：默认只读 kline_all.csv（避免误读其他文件）
    kline_dir = str(RAW_DIR / "kline")
    kline_path = get_kline_main_file(kline_dir)
    
    needs_supplement = False
    df_kline = pd.DataFrame()  # 初始化为空DataFrame
    
    if kline_path and os.path.exists(kline_path):
        try:
            df_kline_all = pd.read_csv(kline_path, dtype={'stock_code': str})
            df_kline = df_kline_all[df_kline_all['instrument'] == symbol].copy()
            
            # 检查数据是否足够（至少需要60天数据用于计算指标）
            if df_kline.empty or len(df_kline) < 60:
                logging.warning(f"{symbol} K线数据不足（少于60天），需要补充")
                needs_supplement = True
            else:
                # 【优化】检查数据新鲜度：判断是否有最近一个交易日的数据
                # 使用 format='ISO8601' 或 format='mixed' 来处理不同的日期格式
                df_kline['date'] = pd.to_datetime(df_kline['date'], format='ISO8601')
                latest_date = df_kline['date'].max()
                latest_trading_date = get_latest_trading_date()
                
                # 【检查1】最新数据日期是否是周末（异常情况）
                if latest_date.weekday() >= 5:
                    logging.warning(f"{symbol} K线数据异常：最新日期 {latest_date.strftime('%Y-%m-%d')} 是周末，需要重新补充")
                    needs_supplement = True
                else:
                    # 【检查2】计算最新数据与最近交易日的差距（天数）
                    days_behind = (latest_trading_date.date() - latest_date.date()).days
                    
                    # 判断当前时间，决定数据新鲜度要求
                    now = datetime.now()
                    is_td = is_trading_day(now)
                    is_after_data_ready = now.hour >= 16
                    
                    # 如果是交易日且16:00后，要求必须有当天数据（严格模式）
                    if is_td and is_after_data_ready:
                        if days_behind > 0:
                            logging.warning(f"{symbol} K线数据已过时（最新: {latest_date.strftime('%Y-%m-%d')}, 应有: {latest_trading_date.strftime('%Y-%m-%d')}, 落后{days_behind}天），需要补充")
                            needs_supplement = True
                        else:
                            logging.info(f"{symbol} K线数据最新（最新: {latest_date.strftime('%Y-%m-%d')}, 应有: {latest_trading_date.strftime('%Y-%m-%d')}, 落后{days_behind}天）")
                    # 其他情况，容忍3天以内的延迟（考虑周末）
                    elif days_behind > 3:
                        logging.warning(f"{symbol} K线数据已过时（最新: {latest_date.strftime('%Y-%m-%d')}, 应有: {latest_trading_date.strftime('%Y-%m-%d')}, 落后{days_behind}天），需要补充")
                        needs_supplement = True
                    else:
                        logging.info(f"{symbol} K线数据最新（最新: {latest_date.strftime('%Y-%m-%d')}, 应有: {latest_trading_date.strftime('%Y-%m-%d')}, 落后{days_behind}天）")
                    
        except Exception as e:
            logging.warning(f"检查K线数据失败: {e}")
            needs_supplement = True
    else:
        # 文件不存在或路径无效
        logging.warning(f"{symbol} K线数据文件不存在，需要补充")
        needs_supplement = True
    
    if needs_supplement:
        logging.info(f"开始补充 {symbol} 的数据...")
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # 智能计算起始日期：如果有历史数据，从最新日期开始；否则补充最近90天
            if not df_kline.empty:
                df_kline['date'] = pd.to_datetime(df_kline['date'], format='ISO8601')
                latest_date = df_kline['date'].max()
                # 从最新日期的前10天开始（有重叠，确保数据连续性）
                start_date = (latest_date - timedelta(days=10)).strftime('%Y-%m-%d')
                logging.info(f"从 {start_date} 开始补充数据（已有数据到 {latest_date.strftime('%Y-%m-%d')}）")
            else:
                # 如果没有历史数据，补充最近90天
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                logging.info(f"没有历史数据，补充最近90天数据")
            
            # 补充K线数据（仅当前股票）
            supplement_service.supplement_kline_data(
                stock_codes=[stock_code],
                start_date=start_date,
                end_date=end_date
            )
            
            # 【优化】龙虎榜数据在main函数中统一补充，不在这里调用
            # 这样可以避免批量分析时，每个股票都重复爬取龙虎榜数据
            
            logging.info(f"{symbol} K线数据补充完成")
            
        except Exception as e:
            logging.error(f"补充 {symbol} K线数据失败: {e}")
    
    # 【新增】检查筹码数据
    chips_path = str(RAW_DIR / "chips_all.csv")
    needs_chips_supplement = False
    
    if os.path.exists(chips_path):
        try:
            chips_df = pd.read_csv(chips_path)
            stock_chips = chips_df[chips_df['instrument'] == symbol].copy()
            
            if stock_chips.empty:
                logging.warning(f"{symbol} 无筹码数据，需要补充")
                needs_chips_supplement = True
            else:
                # 检查筹码数据是否过期（超过7天）
                stock_chips['date'] = pd.to_datetime(stock_chips['date'], format='ISO8601')
                latest_chips_date = stock_chips['date'].max()
                days_behind = (datetime.now().date() - latest_chips_date.date()).days
                
                if days_behind > 7:
                    logging.warning(f"{symbol} 筹码数据已过期（最新: {latest_chips_date.strftime('%Y-%m-%d')}, 落后{days_behind}天），需要补充")
                    needs_chips_supplement = True
                else:
                    logging.info(f"{symbol} 筹码数据最新（最新: {latest_chips_date.strftime('%Y-%m-%d')}）")
        except Exception as e:
            logging.warning(f"检查筹码数据失败: {e}")
            needs_chips_supplement = True
    else:
        logging.warning(f"{symbol} 筹码数据文件不存在，需要补充")
        needs_chips_supplement = True
    
    if needs_chips_supplement:
        logging.info(f"开始补充 {symbol} 的筹码数据...")
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            # 仅补充当前股票的筹码数据
            supplement_service.supplement_chips_data(
                stock_codes=[stock_code],
                start_date=start_date,
                end_date=end_date
            )
            logging.info(f"[OK] {symbol} 筹码数据补充完成")
            
        except Exception as e:
            logging.error(f"补充 {symbol} 筹码数据失败: {e}")
    
    return needs_supplement or needs_chips_supplement


async def check_and_supplement_data_async(symbol, supplement_service):
    """异步检查并补充数据"""
    return await check_and_supplement_data(symbol, supplement_service)


def run_prediction(symbol, supplement_service=None, save_results=True, forced_trigger_date: str | None = None, limit_up_context: dict | None = None):
    """
    运行单个股票的预测
    
    Args:
        symbol: 股票代码（如"300034"或"300034.SZ"）
        supplement_service: Layer3DataSupplement实例（可选）
        save_results: 是否保存结果（默认True）
    
    Returns:
        dict: 预测结果，包含scores和conclusion
    """
    if not symbol.endswith('.SH') and not symbol.endswith('.SZ'):
        if symbol.startswith('6'):
            symbol += '.SH'
        else:
            symbol += '.SZ'
    
    # 检查并补充数据
    if supplement_service:
        try:
            # asyncio.get_event_loop 在新版本 Python 中会产生 DeprecationWarning，这里用 asyncio.run 更稳
            asyncio.run(check_and_supplement_data(symbol, supplement_service))
        except Exception as e:
            logging.warning(f"数据补充失败，继续使用现有数据: {e}")
    
    df_kline, df_chips, df_dragon = fetch_local_data(symbol)
    if df_kline is None:
        logging.error(f"无法加载 {symbol} 的数据")
        return None
    
    df_processed = process_kline(df_kline)
    
    # 寻找最近的触发日 (涨停或大涨)；批量模式下可强制对齐到“涨停池目标日”
    if forced_trigger_date:
        trigger_date = pd.to_datetime(forced_trigger_date, format='ISO8601')
        logging.info(f"【批量对齐】使用指定 trigger_date: {trigger_date.strftime('%Y-%m-%d')}")
    else:
        triggers = df_processed[df_processed['is_limit_up'] | (df_processed['pct_change'] > 7)].tail(5)
        if triggers.empty:
            logging.warning("最近未发现显著异动日，将使用最后一个交易日作为参考")
            trigger_date = df_processed['date'].iloc[-1]
        else:
            trigger_date = triggers['date'].iloc[-1]
        
    logging.info(f"参考触发日期: {trigger_date}")
    
    # 保存临时数据供 PatternMatcher 使用
    # 【优化】使用固定的临时文件名，避免文件累积
    temp_processed_dir = str(TEMP_DIR)
    os.makedirs(temp_processed_dir, exist_ok=True)
    
    # 使用固定文件名，每次覆盖（而不是每个股票一个文件）
    kline_temp_path = os.path.join(temp_processed_dir, 'kline_processed_temp.csv')
    df_processed.to_csv(kline_temp_path, index=False)
    
    # 【优化】获取最新的筹码和龙虎榜数据文件路径
    chips_dir = str(RAW_DIR)
    dragon_dir = str(RAW_DIR / "limit_up")
    
    latest_chips_path = get_latest_file(chips_dir, 'chips*.csv')
    latest_dragon_path = get_latest_file(dragon_dir, 'dragon_list*.csv')
    
    # 如果没有找到，使用默认文件名（向后兼容）
    if not latest_chips_path:
        latest_chips_path = str(RAW_DIR / "chips_all.csv")
    if not latest_dragon_path:
        latest_dragon_path = str(RAW_DIR / "limit_up" / "dragon_list.csv")
    
    # 初始化 PatternMatcher
    base_dir = str(REPO_ROOT)
    matcher = PatternMatcher(
        db_path=os.path.join(base_dir, 'data/historical_patterns.db'),
        processed_kline=kline_temp_path,
        chips_path=latest_chips_path,
        dragon_list_path=latest_dragon_path,
        model_path=os.path.join(base_dir, 'models/second_wave_lgb.model'),
        feature_names_path=os.path.join(base_dir, 'models/feature_names.json'),
        enhanced_features_dir=os.path.join(base_dir, 'data/raw')  # 增强特征数据目录
    )
    
    features = matcher.get_stock_features(symbol, trigger_date, extra_context=limit_up_context)
    if not features:
        logging.error("特征提取失败")
        return None
    
    logging.info(f"提取特征: {features}")
    similar_cases = matcher.find_similar_cases(features)
    scores = matcher.calculate_scores(features, similar_cases)
    
    # 生成报告
    print("\n" + "="*50)
    print(f"股票二波潜力预测报告: {symbol}")
    print(f"分析参考日期 (最近异动日): {trigger_date.strftime('%Y-%m-%d')}")
    print("-" * 50)
    print(f"1. 基础特征:")
    print(f"   - 涨跌幅: {features['pct_change']:.2f}%")
    print(f"   - 量比: {features['volume_ratio']:.2f}")
    print(f"   - RSI: {features['rsi']:.2f}")
    print(f"   - 技术位置: {features['pre_position']}")
    
    print(f"\n2. 资金面特征:")
    print(f"   - 获利盘比例: {features['win_percent']*100:.1f}%")
    print(f"   - 筹码集中度: {features['concentration']:.4f}")
    print(f"   - 龙虎榜净买入: {features['net_buy_amount']/10000:.2f} 万")

    # 【新增】板质量特征（来自涨停池；单股模式可能为空）
    print(f"\n2.1 板质量特征:")
    print(f"   - 首次封板: {features.get('board_first_seal_time', '') or '无'}")
    print(f"   - 最后封板: {features.get('board_last_seal_time', '') or '无'}")
    print(f"   - 炸板次数: {int(features.get('board_explosion_count', 0) or 0)}")
    print(f"   - 封板资金: {float(features.get('board_seal_funds', 0.0) or 0.0)/1e8:.2f} 亿")
    print(f"   - 板质量分: {float(features.get('board_quality_score', 0.0) or 0.0):.1f}/100")

    # 【新增】短周期板结构（A2）
    print(f"\n2.2 短周期板结构（A2）:")
    print(f"   - 近5日涨停次数: {int(features.get('limit_up_count_5d', 0) or 0)}")
    print(f"   - 近10日涨停次数: {int(features.get('limit_up_count_10d', 0) or 0)}")
    print(f"   - 近10日最长连板: {int(features.get('max_consecutive_limit_up_10d', 0) or 0)}")
    print(f"   - 距上次涨停(有效交易日): {int(features.get('days_since_last_limit_up', 999) or 999)}")
    print(f"   - 最近两次涨停间隔(有效交易日): {int(features.get('gap_days_between_limit_ups', 999) or 999)}")

    # 【新增】A3 主线题材匹配（TopN概念命中）
    print(f"\n2.3 主线题材匹配（A3）:")
    print(f"   - 命中Top概念数: {int(features.get('theme_hit_count_topN', 0) or 0)}")
    print(f"   - 最佳概念排名: {int(features.get('theme_best_rank', 999) or 999)}")
    print(f"   - 最佳概念涨幅: {float(features.get('theme_best_gain', 0.0) or 0.0):.2f}%")

    # 【新增】A5 资金结构（龙虎榜席位明细：同花顺 market/longhu，已过滤转债）
    print(f"\n2.4 资金结构（A5 龙虎榜席位明细）:")
    for wd in (1, 3):
        has_data = float(features.get(f'seat_has_data_{wd}d', 0.0) or 0.0)
        if has_data <= 0:
            print(f"   - {wd}日: 无席位明细数据")
            continue
        net_yuan = float(features.get(f'seat_total_net_yuan_{wd}d', 0.0) or 0.0)
        turnover_yuan = float(features.get(f'seat_total_turnover_yuan_{wd}d', 0.0) or 0.0)
        net_to_turnover = float(features.get(f'seat_net_to_turnover_{wd}d', 0.0) or 0.0)
        top1 = float(features.get(f'seat_buy_top1_ratio_{wd}d', 0.0) or 0.0)
        top5 = float(features.get(f'seat_buy_top5_ratio_{wd}d', 0.0) or 0.0)
        inst = float(features.get(f'seat_buy_inst_ratio_{wd}d', 0.0) or 0.0)
        hotm = float(features.get(f'seat_buy_hot_money_ratio_{wd}d', 0.0) or 0.0)
        overlap_cnt = float(features.get(f'seat_buy_sell_overlap_count_{wd}d', 0.0) or 0.0)
        overlap_ratio = float(features.get(f'seat_buy_sell_overlap_ratio_{wd}d', 0.0) or 0.0)
        score = float(features.get(f'seat_structure_score_{wd}d', 0.0) or 0.0)
        print(
            f"   - {wd}日: 成交额 {turnover_yuan/1e8:.2f} 亿 | 净额 {net_yuan/1e8:.2f} 亿 | 净/成 {net_to_turnover:.2f} | "
            f"Top1 {top1:.2f} | Top5 {top5:.2f} | 机构 {inst:.2f} | 游资 {hotm:.2f} | "
            f"买卖重合 {overlap_cnt:.0f}/5({overlap_ratio:.2f}) | 结构分 {score:.1f}/100"
        )
    
    # 【新增】增强特征显示
    print(f"\n3. 增强特征:")
    
    # 3.1 热度特征
    hot_rank = features.get('hot_rank', 999)
    is_hot_stock = features.get('is_hot_stock', False)
    hot_duration = features.get('hot_duration', 0)
    if is_hot_stock or hot_rank <= 200:
        # 注意：Windows 部分终端编码可能为 GBK，避免使用 emoji/特殊符号导致 UnicodeEncodeError
        print(f"   个股热度: 排名 {hot_rank}{' [HOT]' if is_hot_stock else ''}")
        if hot_duration > 0:
            print(f"      持续热度: {hot_duration} 天")
    else:
        print(f"   个股热度: 排名 {hot_rank} (非热门)")
    
    # 3.2 概念特征
    concept_count = features.get('concept_count', 0)
    main_concept_gain = features.get('main_concept_gain', 0.0)
    is_concept_leader = features.get('is_concept_leader', False)
    if concept_count > 0:
        print(f"   概念板块: {concept_count} 个概念")
        print(f"      主概念涨幅: {main_concept_gain:.2f}%")
        if is_concept_leader:
            print("      [LEADER] 概念龙头")
    else:
        print(f"   概念板块: 无数据")
    
    # 3.3 竞价特征
    auction_strength = features.get('auction_strength', 0)
    auction_volume_ratio = features.get('auction_volume_ratio', 0)
    auction_price_gap = features.get('auction_price_gap', 0.0)
    if auction_strength > 60:
        print(f"   竞价表现: 强度 {auction_strength:.0f}/100 [STRONG]")
        print(f"      竞价量比: {auction_volume_ratio:.2f}")
        print(f"      竞价高开: {auction_price_gap:.2f}%")
    elif auction_strength > 30:
        print(f"   竞价表现: 强度 {auction_strength:.0f}/100 (中等)")
    else:
        print(f"   竞价表现: 强度 {auction_strength:.0f}/100 (弱)")
    
    # 3.4 市场情绪
    is_hot_sector = features.get('is_hot_sector', False)
    sector_rank = features.get('sector_rank', 999)
    market_sentiment_score = features.get('market_sentiment_score', 50.0)
    market_sentiment_level_label = str(features.get('market_sentiment_level_label', '') or '')
    if is_hot_sector:
        print(f"   所属板块: 热门板块 (排名 {sector_rank})")
    if market_sentiment_level_label:
        print(f"   市场情绪: {market_sentiment_score:.1f}/100（{market_sentiment_level_label}）")
    else:
        print(f"   市场情绪得分: {market_sentiment_score:.1f}/100")
    print(f"   情绪趋势: 2D {float(features.get('sentiment_slope_2d', 0.0) or 0.0):+.1f}, 3D {float(features.get('sentiment_slope_3d', 0.0) or 0.0):+.1f}")
    print(f"   情绪回退天数: {int(features.get('sentiment_fallback_days', 0) or 0)}")

    # 上证指数环境（可选）
    if bool(features.get('index_has_data', False)):
        try:
            print(
                f"   上证环境: ret1d {float(features.get('index_ret_1d', 0.0) or 0.0):+.2f}% | "
                f"ret5d {float(features.get('index_ret_5d', 0.0) or 0.0):+.2f}% | "
                f"MA5乖离 {float(features.get('index_bias_ma5', 0.0) or 0.0)*100:+.2f}% | "
                f"量能比 {float(features.get('index_amount_ratio_5d', 1.0) or 1.0):.2f} | "
                f"连跌 {int(features.get('index_down_streak', 0) or 0)}"
            )
        except Exception:
            pass
    
    print(f"\n4. 评分结果:")
    print(f"   - 强度分: {scores['strength_score']:.1f}")
    print(f"   - 规则概率: {scores['rule_prob']:.3f}")
    print(f"   - ML 概率: {scores['ml_prob']:.3f}")
    print(f"   - 综合概率: {scores['final_prob']:.3f}")
    
    print(f"\n5. 模式匹配:")
    print(f"   - 相似案例数: {scores['sample_size']}")
    print(f"   - 历史成功率: {scores['pattern_success_rate']*100:.1f}%")
    print(f"   - 历史平均最大涨幅: {scores['avg_max_return']*100:.1f}%")
    
    print("-" * 50)
    if scores['final_prob'] > 0.6:
        conclusion = "该股具备极高的二波潜力，建议重点关注。"
        print(f"结论: {conclusion}")
    elif scores['final_prob'] > 0.5:
        conclusion = "具备一定潜力，建议谨慎观察。"
        print(f"结论: {conclusion}")
    else:
        conclusion = "潜力一般，建议继续等待更明确信号。"
        print(f"结论: {conclusion}")
    print("="*50 + "\n")
    
    result = {
        'symbol': symbol,
        'trigger_date': trigger_date.strftime('%Y-%m-%d'),
        'features': features,
        'scores': scores,
        'conclusion': conclusion,
        'final_prob': scores['final_prob']
    }
    
    # 只保存高潜力和中等潜力的结果
    if save_results and scores['final_prob'] > 0.5:
        save_prediction_result(result)
    
    return result


def save_prediction_result(result):
    """
    保存预测结果到JSON文件
    
    文件命名规则：按trigger_date（异动日期）组织
    - predictions_trigger_20260109.json: 2026-01-09涨停的股票分析
    
    去重规则：
    - 同一个trigger_date下，同一个股票只保存一次（避免重复分析）
    - 不同trigger_date的分析会分别保存（连板股票会有多个分析）
    """
    results_dir = str(RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)
    
    # 【优化1】使用trigger_date（异动日期）作为文件名，而不是分析日期
    trigger_date = result.get('trigger_date', '')
    if not trigger_date:
        # 如果没有trigger_date，使用当前日期作为fallback
        trigger_date = datetime.now().strftime('%Y-%m-%d')
    
    # 将日期格式化为YYYYMMDD
    trigger_date_str = trigger_date.replace('-', '')
    results_file = os.path.join(results_dir, f'predictions_trigger_{trigger_date_str}.json')
    
    # 读取已有结果
    results = []
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception as e:
            logging.warning(f"读取已有结果失败: {e}")
    
    # 【优化2】检查是否已存在该股票的结果（同一trigger_date下去重）
    found = False
    for i, r in enumerate(results):
        if r.get('symbol') == result['symbol']:
            # 如果已存在，更新结果
            results[i] = result
            found = True
            logging.info(f"更新已有分析结果: {result['symbol']} (trigger_date={trigger_date})")
            break
    
    if not found:
        results.append(result)
        logging.info(f"添加新分析结果: {result['symbol']} (trigger_date={trigger_date})")
    
    # 保存结果
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logging.info(f"[OK] 结果已保存: {results_file} (共{len(results)}只股票)")
    except Exception as e:
        logging.error(f"保存结果失败: {e}")


def save_predictions_topk_for_date(trigger_date: str, results: list[dict], topk: int) -> str:
    """
    将某个 trigger_date 的结果写成单个文件（只保留 TOPK）。
    输出：results/predictions_trigger_YYYYMMDD.json
    """
    results_dir = str(RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)

    td = (trigger_date or "").strip()
    if not td:
        td = datetime.now().strftime("%Y-%m-%d")
    td_str = td.replace("-", "")
    out_file = os.path.join(results_dir, f"predictions_trigger_{td_str}.json")

    cleaned = [r for r in results if isinstance(r, dict)]
    cleaned.sort(key=lambda r: float(r.get("final_prob", 0.0) or 0.0), reverse=True)
    picked = cleaned[: max(1, int(topk))]

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=2, default=str)

    return out_file


def _parse_yyyy_mm_dd(s: str) -> date_type:
    return datetime.strptime(s, "%Y-%m-%d").date()


def iter_trading_days_inclusive(start_date: str, end_date: str) -> list[date_type]:
    """
    生成 [start_date, end_date] 闭区间内的交易日列表：
    - 优先使用 BigQuant all_trading_days（is_trading_day 内部实现）
    - 失败则回退 weekday<5
    """
    start = _parse_yyyy_mm_dd(start_date)
    end = _parse_yyyy_mm_dd(end_date)
    if start > end:
        start, end = end, start

    out: list[date_type] = []
    cur = start
    while cur <= end:
        if is_trading_day(cur):
            out.append(cur)
        cur = cur + timedelta(days=1)
    return out


def get_limit_up_stocks_for_date(
    trade_date: date_type,
    supplement_service: Layer3DataSupplement,
    return_details: bool = False,
):
    """
    获取指定交易日的涨停池（支持 Moma/智兔 自动切换）。
    返回：
    - stock_codes: list[str] （6位代码）
    - details_map: dict[str, dict]（可选，板质量字段映射）
    - trade_date_formatted: YYYY-MM-DD
    """
    trade_date_str = trade_date.strftime("%Y%m%d")
    trade_date_formatted = trade_date.strftime("%Y-%m-%d")

    def _call_once():
        return supplement_service.api_adapter.get_limit_up_pool(trade_date_str)

    try:
        limit_up_list = _call_once()
    except Exception as e:
        msg = str(e)
        # 配额/限流：切换 API 重试一次
        if supplement_service._is_api_limit_error(msg):  # type: ignore
            try:
                supplement_service.switch_api()
                limit_up_list = _call_once()
            except Exception as e2:
                logging.error(f"涨停池获取失败（已切换API仍失败） {trade_date_formatted}: {e2}")
                limit_up_list = []
        else:
            logging.error(f"涨停池获取失败 {trade_date_formatted}: {e}")
            limit_up_list = []

    if not limit_up_list:
        if return_details:
            return [], {}, trade_date_formatted
        return []

    stock_codes: list[str] = []
    details_map: dict[str, dict] = {}

    for item in limit_up_list:
        if not isinstance(item, dict):
            continue

        # 兼容 MomaAdapter / ZhituAdapter 返回字段
        stock_code = str(item.get("stock_code") or item.get("code") or item.get("dm") or "").strip()
        if not (len(stock_code) == 6 and stock_code.isdigit()):
            continue

        stock_codes.append(stock_code)

        # 统一映射板质量字段（尽量从各种字段里取到）
        details_map[stock_code] = {
            "first_seal_time": item.get("first_seal_time", "") or item.get("fbt", "") or item.get("first_seal", ""),
            "last_seal_time": item.get("last_seal_time", "") or item.get("lbt", "") or item.get("last_seal", ""),
            "explosion_count": item.get("explosion_count", 0) or item.get("zbc", 0) or item.get("explosion", 0),
            "seal_funds": item.get("seal_funds", 0.0) or item.get("zj", 0.0) or item.get("seal_amount", 0.0),
            "turnover_rate": item.get("turnover_rate", 0.0) or item.get("hs", 0.0) or item.get("turnover", 0.0),
            "consecutive_boards": item.get("consecutive_boards", 0) or item.get("lbc", 0),
            "limit_up_statistics": item.get("limit_up_statistics", "") or item.get("tj", ""),
        }

    if return_details:
        return stock_codes, details_map, trade_date_formatted
    return stock_codes


def get_yesterday_limit_up_stocks(force_today=False, return_details: bool = False):
    """
    获取最近一个交易日的涨停股票列表
    
    规则：
    1. 如果今天是交易日（周一~周五）且数据已更新（16:00后）→ 获取今天的涨停股票
    2. 如果今天是交易日但数据未更新（16:00前）→ 获取昨天的涨停股票
    3. 如果今天是周末/节假日 → 获取最近一个交易日的涨停股票
    
    注意：虽然股市15:00收盘，但数据API通常要到16:00左右才完全更新
    
    Args:
        force_today: 强制使用今天的数据（即使在16:00前）
    """
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    target_date = _pick_limit_up_target_date(now, force_today=force_today)
    # 友好日志
    if force_today and is_trading_day(now):
        logging.info(f"【强制模式】获取今天的涨停股票（{current_hour:02d}:{current_minute:02d}）")
    elif is_trading_day(now) and current_hour >= 16:
        logging.info(f"今天是交易日且数据已更新（{current_hour:02d}:{current_minute:02d}），获取今天的涨停股票")
    else:
        logging.info(f"获取最近一个交易日的涨停股票（{current_hour:02d}:{current_minute:02d}）")
    
    trade_date = target_date.strftime('%Y%m%d')
    trade_date_formatted = target_date.strftime('%Y-%m-%d')
    
    logging.info(f"目标日期: {trade_date_formatted}")
    
    try:
        moma_adapter = MomaAdapter()
        limit_up_list = moma_adapter.get_limit_up_pool(trade_date)
        
        if not limit_up_list:
            logging.warning(f"未获取到 {trade_date_formatted} 的涨停股票")
            return []
        
        # 提取股票代码列表 + （可选）涨停池明细映射（用于板质量）
        stock_codes = []
        details_map: dict[str, dict] = {}
        for item in limit_up_list:
            stock_code = item.get('stock_code') or item.get('code') or item.get('dm', '')
            if stock_code and len(stock_code) == 6:
                stock_codes.append(stock_code)
                # 保留关键字段（MomaAdapter 已做字段映射：first_seal_time/last_seal_time/explosion_count/seal_funds 等）
                details_map[stock_code] = {
                    'first_seal_time': item.get('first_seal_time', ''),
                    'last_seal_time': item.get('last_seal_time', ''),
                    'explosion_count': item.get('explosion_count', 0),
                    'seal_funds': item.get('seal_funds', 0.0),
                    'turnover_rate': item.get('turnover_rate', 0.0),
                    'consecutive_boards': item.get('consecutive_boards', item.get('lbc', 0)),
                    'limit_up_statistics': item.get('limit_up_statistics', item.get('tj', '')),
                }
        
        logging.info(f"获取到 {len(stock_codes)} 只涨停股票")
        if return_details:
            return stock_codes, details_map, trade_date_formatted
        return stock_codes
    
    except Exception as e:
        logging.error(f"获取涨停股票列表失败: {e}")
        return []


def get_analyzed_stocks(trigger_date):
    """
    获取指定trigger_date已经分析过的股票列表
    
    Args:
        trigger_date: 异动日期（格式：YYYY-MM-DD）
    
    Returns:
        已分析的股票代码集合
    """
    results_dir = str(RESULTS_DIR)
    trigger_date_str = trigger_date.replace('-', '')
    results_file = os.path.join(results_dir, f'predictions_trigger_{trigger_date_str}.json')
    
    analyzed_stocks = set()
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                for r in results:
                    symbol = r.get('symbol', '')
                    # 提取股票代码（去掉市场后缀）
                    if '.' in symbol:
                        stock_code = symbol.split('.')[0]
                        analyzed_stocks.add(stock_code)
                    else:
                        analyzed_stocks.add(symbol)
        except Exception as e:
            logging.warning(f"读取已分析股票列表失败: {e}")
    
    return analyzed_stocks


def main():
    """主函数：批量分析涨停股票"""
    parser = argparse.ArgumentParser(description='Predict stock second wave potential.')
    parser.add_argument('--symbol', type=str, help='Single stock symbol (e.g., 300034)')
    parser.add_argument('--batch', action='store_true', help='Batch analyze yesterday limit-up stocks')
    parser.add_argument('--batch-range', action='store_true', help='Batch analyze limit-up stocks in a date range (trading days)')
    parser.add_argument('--start-date', type=str, default=None, help='Range start date YYYY-MM-DD (for --batch-range)')
    parser.add_argument('--end-date', type=str, default=None, help='Range end date YYYY-MM-DD (for --batch-range)')
    parser.add_argument('--topk', type=int, default=3, help='Per-day keep TOPK (for --batch-range)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing daily results files (for --batch-range)')
    parser.add_argument('--force-today', action='store_true', help='Force use today\'s data (even before 16:00)')
    parser.add_argument(
        '--force-refresh-from',
        type=str,
        default=None,
        help="(Optional) Force refresh Kline from a given date (YYYY-MM-DD or YYYYMMDD) in batch pre-supplement step.",
    )
    args = parser.parse_args()
    
    # 初始化数据补充服务
    supplement_service = Layer3DataSupplement(use_moma=True)

    if args.batch_range:
        if not args.start_date or not args.end_date:
            raise SystemExit("--batch-range requires --start-date and --end-date (YYYY-MM-DD)")

        start_date = str(args.start_date)
        end_date = str(args.end_date)
        topk = max(1, int(args.topk))
        overwrite = bool(args.overwrite)

        trading_days = iter_trading_days_inclusive(start_date, end_date)
        logging.info("=" * 80)
        logging.info(f"区间批量模式：{start_date} ~ {end_date}（交易日={len(trading_days)}，TOPK={topk}）")
        logging.info(f"写入目录：{RESULTS_DIR}")
        logging.info("=" * 80)

        for d in trading_days:
            trigger_date = d.strftime("%Y-%m-%d")
            trigger_date_str = trigger_date.replace("-", "")
            out_file = Path(RESULTS_DIR) / f"predictions_trigger_{trigger_date_str}.json"

            if out_file.exists() and (not overwrite):
                logging.info(f"[SKIP] {trigger_date}: already exists -> {out_file}")
                continue

            # 1) 获取当日涨停池（支持API自动切换）
            stock_codes, limit_up_details, trade_date_formatted = get_limit_up_stocks_for_date(
                d, supplement_service, return_details=True
            )
            if not stock_codes:
                logging.warning(f"[EMPTY] {trigger_date}: 无涨停池数据，跳过")
                continue

            logging.info("=" * 80)
            logging.info(f"[DAY] {trigger_date}: 涨停池 {len(stock_codes)} 只")
            logging.info("=" * 80)

            # 2) 预补K线到 trigger_date（若本地已有更全数据，会自动跳过）
            try:
                target_date = pd.to_datetime(trigger_date, format="ISO8601")
                end_date_kline = trigger_date
                start_date_kline = (target_date.to_pydatetime() - timedelta(days=120)).strftime("%Y-%m-%d")
                supplement_service.supplement_kline_data(
                    stock_codes=stock_codes,
                    start_date=start_date_kline,
                    end_date=end_date_kline,
                    force_refresh_from=args.force_refresh_from,
                )
            except Exception as e:
                logging.warning(f"⚠️ {trigger_date}: K线预补失败: {e}（将继续按个股分析时按需补充）")

            # 3) 龙虎榜：按 trigger_date 的近30天窗口补一次（本地已有更全时会自动跳过）
            try:
                end_d = trigger_date
                start_d = (pd.to_datetime(trigger_date, format="ISO8601") - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                supplement_service.supplement_dragon_list_data(start_date=start_d, end_date=end_d)
            except Exception as e:
                logging.warning(f"⚠️ {trigger_date}: 龙虎榜补充失败: {e}")

            # 4) 批量分析（不走 save_prediction_result，最后统一写 TOPK 文件）
            results: list[dict] = []
            for i, stock_code in enumerate(stock_codes, 1):
                logging.info(f"[{trigger_date}] [{i}/{len(stock_codes)}] 分析股票: {stock_code}")
                try:
                    r = run_prediction(
                        stock_code,
                        supplement_service=supplement_service,
                        save_results=False,
                        forced_trigger_date=trigger_date,
                        limit_up_context=limit_up_details.get(stock_code),
                    )
                    if isinstance(r, dict):
                        results.append(r)
                except Exception as e:
                    logging.error(f"{trigger_date} 分析 {stock_code} 失败: {e}")
                    continue

            if not results:
                logging.warning(f"[EMPTY] {trigger_date}: 无有效预测结果，跳过写文件")
                continue

            # 5) 写入当日 TOPK
            out_path = save_predictions_topk_for_date(trigger_date, results, topk=topk)

            # 日内摘要
            results.sort(key=lambda r: float(r.get("final_prob", 0.0) or 0.0), reverse=True)
            summary = ", ".join([f"{r.get('symbol')}({float(r.get('final_prob', 0.0) or 0.0):.3f})" for r in results[:topk]])
            logging.info(f"[OK] {trigger_date}: saved TOP{topk} -> {out_path}")
            logging.info(f"[TOP{topk}] {trigger_date}: {summary}")

        logging.info("[DONE] 区间批量完成")
        return

    if args.batch:
        # 批量分析模式：获取涨停股票并逐个分析
        stock_codes, limit_up_details, trade_date_formatted = get_yesterday_limit_up_stocks(
            force_today=args.force_today,
            return_details=True,
        )
        
        if not stock_codes:
            logging.error("未获取到涨停股票列表，退出")
            return
        
        # 【优化3】trigger_date 与涨停股票列表日期严格对齐
        trigger_date = trade_date_formatted
        target_date = pd.to_datetime(trigger_date, format='ISO8601')
        
        # 检查已分析的股票
        analyzed_stocks = get_analyzed_stocks(trigger_date)
        
        if analyzed_stocks:
            logging.info(f"已分析股票数: {len(analyzed_stocks)} 只，将跳过这些股票")
            # 过滤出未分析的股票
            stock_codes_to_analyze = [code for code in stock_codes if code not in analyzed_stocks]
            skipped_count = len(stock_codes) - len(stock_codes_to_analyze)
            if skipped_count > 0:
                logging.info(f"跳过 {skipped_count} 只已分析的股票")
            stock_codes = stock_codes_to_analyze
            # 同步过滤板质量明细
            limit_up_details = {c: limit_up_details.get(c, {}) for c in stock_codes}
        
        if not stock_codes:
            logging.info(f"所有股票（{len(analyzed_stocks)}只）均已分析，无需重复分析")
            return

        # 【新增】批量分析前，优先把“本次涨停名单”的K线补齐到 trigger_date（只补名单，不做全市场）
        # 目的：让当日/近5日 volume、volume_ratio、MA 等计算更稳定，避免出现 volume 全0 → 量比 NaN 的问题
        logging.info("=" * 80)
        logging.info("批量分析准备：预补K线数据（仅涨停名单，增量更新到 trigger_date）")
        logging.info("=" * 80)
        try:
            # 使用 trigger_date 对齐本次分析日期（而不是用“现在时间”的 end_date）
            end_date_kline = trigger_date
            # 回看窗口：至少覆盖 60 日窗口 + 20 日趋势 + 一点冗余
            start_date_kline = (target_date.to_pydatetime() - timedelta(days=120)).strftime('%Y-%m-%d')
            # 默认不强制刷新（避免每次批量都下载刷新）；如有需要可通过 --force-refresh-from 显式开启
            force_refresh_from = args.force_refresh_from

            supplement_service.supplement_kline_data(
                stock_codes=stock_codes,
                start_date=start_date_kline,
                end_date=end_date_kline,
                force_refresh_from=force_refresh_from,
            )
            logging.info("[OK] K线数据预补完成")
        except Exception as e:
            logging.warning(f"⚠️ K线数据预补失败: {e}，将继续按个股分析时再按需补充")
        
        # 【优化】批量分析前，统一补充一次龙虎榜数据（避免每个股票都重复下载）
        logging.info("=" * 80)
        logging.info("批量分析准备：统一补充龙虎榜数据（使用 BigQuant API）")
        logging.info("=" * 80)
        try:
            # 补充最近30天的龙虎榜数据（覆盖大部分股票的需求）
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # 直接调用（不再需要asyncio，因为改用API）
            supplement_service.supplement_dragon_list_data(
                start_date=start_date,
                end_date=end_date
            )
            logging.info("[OK] 龙虎榜数据补充完成")
        except Exception as e:
            logging.warning(f"⚠️ 龙虎榜数据补充失败: {e}，将在个股分析时按需补充")
        
        logging.info("=" * 80)
        logging.info(f"开始批量分析 {len(stock_codes)} 只涨停股票 (trigger_date={trigger_date})")
        logging.info("=" * 80)
        
        results = []
        for i, stock_code in enumerate(stock_codes, 1):
            logging.info(f"\n[{i}/{len(stock_codes)}] 分析股票: {stock_code}")
            try:
                result = run_prediction(
                    stock_code,
                    supplement_service=supplement_service,
                    save_results=True,
                    forced_trigger_date=trigger_date,
                    limit_up_context=limit_up_details.get(stock_code),
                )
                if result:
                    results.append(result)
            except Exception as e:
                logging.error(f"分析 {stock_code} 失败: {e}")
                continue
        
        # 统计结果
        high_potential = [r for r in results if r['final_prob'] > 0.6]
        medium_potential = [r for r in results if 0.5 < r['final_prob'] <= 0.6]
        
        print("\n" + "="*80)
        print("批量分析完成")
        print("="*80)
        print(f"总计分析: {len(results)} 只股票")
        print(f"极高潜力 (>0.6): {len(high_potential)} 只")
        print(f"中等潜力 (0.5-0.6): {len(medium_potential)} 只")
        
        if high_potential:
            print("\n极高潜力股票:")
            for r in sorted(high_potential, key=lambda x: x['final_prob'], reverse=True):
                print(f"  {r['symbol']}: 概率={r['final_prob']:.3f}")
        
        if medium_potential:
            print("\n中等潜力股票:")
            for r in sorted(medium_potential, key=lambda x: x['final_prob'], reverse=True):
                print(f"  {r['symbol']}: 概率={r['final_prob']:.3f}")
        print("="*80)
        
    elif args.symbol:
        # 单股票分析模式
        logging.info("=" * 80)
        logging.info(f"单股票分析：{args.symbol}")
        logging.info("=" * 80)
        
        # 【优化】单股票分析前，补充一次龙虎榜数据
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # 直接调用（不再需要asyncio）
            supplement_service.supplement_dragon_list_data(
                start_date=start_date,
                end_date=end_date
            )
            logging.info("[OK] 龙虎榜数据补充完成")
        except Exception as e:
            logging.warning(f"⚠️ 龙虎榜数据补充失败: {e}")
        
        run_prediction(args.symbol, supplement_service=supplement_service, save_results=True)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
