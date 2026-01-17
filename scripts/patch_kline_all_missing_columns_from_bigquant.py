"""
在现有 kline_all.csv 基础上“补列”（更省流量版）

目标：
- 你的现有 kline_all.csv 只有 9 列：low/date/adjust_factor/volume/high/close/open/instrument/amount
- BigQuant cn_stock_bar1d 还提供 7 列：name/pre_close/deal_number/change_ratio/turn/upper_limit/lower_limit
- 本脚本只下载缺失列（不重下已存在列），并按 (date, instrument) 合并回 kline_all.csv

特点：
- 更省流量：只下载缺失列（+ key 列）
- 更稳：按日期分批下载（避免单次数据过大）
- 更省内存：读取本地 kline_all.csv 分块 join，再写临时文件；不会把“本地+补列”全量同时 merge 到一个大 DataFrame

用法：
cd TradingAgents-chinese-market/AlphaSignal-CN
python scripts/patch_kline_all_missing_columns_from_bigquant.py --start-date 2023-01-01 --end-date 2026-01-14 --batch-days 120

默认：
- input/output: data/raw/kline/kline_all.csv（会先备份旧文件，再原子替换）

注意：
- 需要你下周 BigQuant 流量恢复后再运行
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _import_dai():
    try:
        from bigquantdai import dai  # type: ignore
        return dai
    except Exception:
        try:
            from bigquant.api import dai  # type: ignore
            return dai
        except Exception:
            import dai  # type: ignore
            return dai


def _date_batches(start_date: str, end_date: str, batch_days: int) -> List[Tuple[str, str]]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    batches: List[Tuple[str, str]] = []
    cur = start
    while cur <= end:
        nxt = min(end, cur + timedelta(days=batch_days - 1))
        batches.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + timedelta(days=1)
    return batches


def _standardize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    输出列顺序尽量对齐 BigQuant cn_stock_bar1d 的列顺序（你提供的那16列）
    """
    desired = [
        "date",
        "instrument",
        "name",
        "adjust_factor",
        "pre_close",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "deal_number",
        "amount",
        "change_ratio",
        "turn",
        "upper_limit",
        "lower_limit",
    ]
    cols = [c for c in desired if c in df.columns] + [c for c in df.columns if c not in desired]
    return df[cols]


def _load_name_map(dai, ref_date: str) -> Dict[str, str]:
    """
    name 是静态字段，没必要按全量日期下载。
    用 end_date 当天（或最近交易日）抓一批 instrument->name 映射即可。
    """
    sql = f"""
    SELECT instrument, name
    FROM cn_stock_bar1d
    WHERE date = '{ref_date}'
    """
    df = dai.query(sql, full_db_scan=True).df()
    if df is None or df.empty:
        return {}
    df = df.drop_duplicates(subset=["instrument"], keep="last")
    return dict(zip(df["instrument"].astype(str), df["name"].astype(str)))


def patch_kline_all(
    kline_path: Path,
    start_date: str,
    end_date: str,
    batch_days: int,
    chunksize: int,
) -> None:
    if not kline_path.exists():
        raise FileNotFoundError(f"未找到本地K线文件: {kline_path}")

    dai = _import_dai()

    # 备份与临时输出
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = kline_path.with_name(f"{kline_path.stem}.tmp_{ts}{kline_path.suffix}")
    bak_path = kline_path.with_name(f"{kline_path.stem}.bak_{ts}{kline_path.suffix}")

    print(f"[INFO] 输入文件: {kline_path}")
    print(f"[INFO] 临时文件: {tmp_path}")
    print(f"[INFO] 备份文件: {bak_path}")

    # 下载 name 映射（小表）
    name_map = _load_name_map(dai, ref_date=end_date)
    print(f"[INFO] name_map size: {len(name_map)}")

    # 分批下载“缺失列”（不含 name）
    patch_cols = [
        "date",
        "instrument",
        "pre_close",
        "deal_number",
        "change_ratio",
        "turn",
        "upper_limit",
        "lower_limit",
    ]

    patch_parts: List[pd.DataFrame] = []
    for b_start, b_end in _date_batches(start_date, end_date, batch_days=batch_days):
        print(f"[INFO] 下载补列批次: {b_start} ~ {b_end}")
        sql = f"""
        SELECT {", ".join(patch_cols)}
        FROM cn_stock_bar1d
        WHERE date >= '{b_start}' AND date <= '{b_end}'
        """
        df = dai.query(sql, full_db_scan=True).df()
        if df is None or df.empty:
            print("[WARN] 批次为空，跳过")
            continue

        # dtype 压缩
        df["date"] = pd.to_datetime(df["date"], format="ISO8601")
        df["instrument"] = df["instrument"].astype(str)
        for c in ["pre_close", "change_ratio", "turn", "upper_limit", "lower_limit"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        if "deal_number" in df.columns:
            df["deal_number"] = pd.to_numeric(df["deal_number"], errors="coerce").astype("Int32")

        patch_parts.append(df)
        print(f"[INFO]   批次行数: {len(df)}")

    if not patch_parts:
        raise RuntimeError("未下载到任何补列数据，请检查日期范围。")

    patch_df = pd.concat(patch_parts, ignore_index=True)
    patch_df = patch_df.drop_duplicates(subset=["date", "instrument"], keep="last")
    patch_df["instrument"] = patch_df["instrument"].astype("category")
    patch_df = patch_df.set_index(["date", "instrument"]).sort_index()
    print(f"[INFO] 补列数据总行数(去重后): {len(patch_df)}")

    # 分块读取本地 kline_all.csv，join 补列后写入临时文件
    wrote_header = False
    total_rows = 0

    # 为了 join，保持与 patch_df 一致的 key 类型
    for chunk in pd.read_csv(kline_path, chunksize=chunksize):
        if "date" not in chunk.columns or "instrument" not in chunk.columns:
            raise RuntimeError("kline_all.csv 必须包含 date 和 instrument 列才能补列。")

        chunk["date"] = pd.to_datetime(chunk["date"], format="ISO8601")
        chunk["instrument"] = chunk["instrument"].astype(str).astype("category")

        chunk = chunk.set_index(["date", "instrument"]).sort_index()
        chunk = chunk.join(patch_df, how="left")
        chunk = chunk.reset_index()

        # 补 name（从 name_map 映射）
        if "name" not in chunk.columns:
            chunk["name"] = chunk["instrument"].astype(str).map(name_map)
        else:
            # 若已有 name，但缺失，则补齐
            miss = chunk["name"].isna()
            if miss.any():
                chunk.loc[miss, "name"] = chunk.loc[miss, "instrument"].astype(str).map(name_map)

        chunk = _standardize_output_columns(chunk)

        chunk.to_csv(
            tmp_path,
            mode="a",
            header=not wrote_header,
            index=False,
            encoding="utf-8-sig",
        )
        wrote_header = True
        total_rows += len(chunk)
        print(f"[INFO] 写入 {len(chunk)} 行，累计 {total_rows}")

    # 备份旧文件并替换
    if kline_path.exists():
        kline_path.replace(bak_path)
    tmp_path.replace(kline_path)

    size_mb = kline_path.stat().st_size / 1024 / 1024
    print(f"[OK] 完成补列：{kline_path}（rows={total_rows}, size={size_mb:.2f}MB）")
    print(f"[OK] 旧文件备份：{bak_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kline-path",
        default="data/raw/kline/kline_all.csv",
        help="本地K线文件（默认 data/raw/kline/kline_all.csv）",
    )
    parser.add_argument("--start-date", required=True, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--batch-days", type=int, default=120, help="按日期分批下载天数（默认120）")
    parser.add_argument("--chunksize", type=int, default=200_000, help="本地CSV分块行数（默认200000）")
    args = parser.parse_args()

    patch_kline_all(
        kline_path=Path(args.kline_path),
        start_date=args.start_date,
        end_date=args.end_date,
        batch_days=args.batch_days,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()

