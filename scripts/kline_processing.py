"""
轻量级 K 线处理与技术指标计算（无额外依赖）
由Task 1.2 的check_price_consistency.py 和 eval_offline.py 复用

用途：
- 供推理侧（predict_stock.py）与一致性闸门（check_price_consistency.py）复用
- 避免在导入 predict_stock.py 时被其它可选依赖（如 httpx/playwright 等）阻断

口径（Task 1.2）：
- 本仓库 kline_all.csv 来源为 BigQuant `cn_stock_bar1d`（后复权），open/high/low/close 已是复权口径
- *_qfq 直接等于原价，禁止再次使用 adjust_factor 二次复权
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd


def process_kline(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算技术指标（最小集）：ma5/ma10/ma20/ma60、成交量均线/量比、pct_change、rsi，并标记涨停。

    注意：
    - 仅在“有效交易日”（close_qfq notna 且 volume>0）序列上计算滚动指标，避免停牌日 NaN 扩散。
    - *_qfq 直接等于原价（cn_stock_bar1d 已复权）。
    """
    logging.info("计算技术指标...")

    # 使用 format='ISO8601' 处理不同的日期格式（支持 YYYY-MM-DD 和 YYYY-MM-DD HH:MM:SS）
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="ISO8601", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 统一复权/价格口径（Task 1.2）
    # 注意：如果 *_qfq 字段已存在且有效（说明调用方已做过复权转换），则保留不覆盖
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            # 检查 *_qfq 字段是否已存在且有有效值
            qfq_col = f"{col}_qfq"
            if qfq_col in df.columns and df[qfq_col].notna().any():
                # 已存在有效的前复权价格，保留不覆盖
                pass
            else:
                # 不存在或全为NaN，使用原价（假设已复权）
                df[qfq_col] = df[col]
        else:
            df[f"{col}_qfq"] = np.nan

    # === 处理停牌/无交易日：只在“有效交易日”上计算技术指标 ===
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce")
    df["amount"] = pd.to_numeric(df.get("amount", np.nan), errors="coerce")
    df["close_qfq"] = pd.to_numeric(df.get("close_qfq", np.nan), errors="coerce")

    valid_mask = df["close_qfq"].notna() & (df["volume"].fillna(0) > 0)

    # 先初始化指标列（无效日保持 NaN，不参与触发判断）
    for col in ["ma5", "ma10", "ma20", "ma60", "v_ma5", "volume_ratio", "pct_change", "rsi"]:
        df[col] = np.nan

    df_valid = df.loc[valid_mask].copy()
    if not df_valid.empty:
        # 均线（按“有效交易日序列”滚动）
        df_valid["ma5"] = df_valid["close_qfq"].rolling(window=5).mean()
        df_valid["ma10"] = df_valid["close_qfq"].rolling(window=10).mean()
        df_valid["ma20"] = df_valid["close_qfq"].rolling(window=20).mean()
        df_valid["ma60"] = df_valid["close_qfq"].rolling(window=60).mean()

        # 成交量均线/量比（避免分母为0 -> inf）
        df_valid["v_ma5"] = df_valid["volume"].rolling(window=5).mean()
        denom = df_valid["v_ma5"].shift(1)
        df_valid["volume_ratio"] = np.where(denom > 0, df_valid["volume"] / denom, np.nan)

        # 涨跌幅（关闭 fill_method 以消除 FutureWarning）
        df_valid["pct_change"] = df_valid["close_qfq"].pct_change(fill_method=None) * 100

        # RSI（有效交易日序列）
        delta = df_valid["close_qfq"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df_valid["rsi"] = 100 - (100 / (1 + rs))

        # 回写到原 df（按 index 对齐）
        for col in ["ma5", "ma10", "ma20", "ma60", "v_ma5", "volume_ratio", "pct_change", "rsi"]:
            df.loc[df_valid.index, col] = df_valid[col]

    # 标记涨停（根据板块不同设置不同阈值）
    instrument = str(df["instrument"].iloc[0]) if "instrument" in df.columns and not df.empty else ""
    stock_code = instrument.split(".")[0] if "." in instrument else instrument

    # 判断涨停阈值
    if stock_code.startswith(("92", "43", "8")):
        # 北交所：30%涨停
        limit = 29.5
    elif stock_code.startswith(("300", "301", "688", "689")):
        # 创业板、科创板：20%涨停
        limit = 19.5
    else:
        # 主板（沪深）：10%涨停
        limit = 9.5

    df["is_limit_up"] = (df["pct_change"] >= limit) & df["pct_change"].notna()
    return df

