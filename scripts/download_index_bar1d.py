"""
下载指数日行情（cn_stock_index_bar1d）到本地 data/raw/index_bar1d.csv（增量）。

默认仅下载上证指数（000001.SH），用于短线环境因子：
- 连续涨跌、3/5日涨跌
- MA5/10/20 位置、乖离
- 成交额相对量能（当日/近5日均额）

说明：
- 该脚本需要 BigQuant SDK（bigquantdai / bigquant.api / dai）。
- 若本机未安装，会直接抛错，避免“静默失败”。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def _import_bigquant_dai():
    try:
        from bigquantdai import dai  # type: ignore
        return dai
    except ImportError:
        try:
            from bigquant.api import dai  # type: ignore
            return dai
        except ImportError:
            try:
                import dai  # type: ignore
                return dai
            except ImportError as e:
                raise ImportError(
                    "BigQuant SDK 不可用：缺少 bigquantdai/bigquant.api/dai 模块，请先安装并配置。"
                ) from e


def _read_existing(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"instrument": str})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="ISO8601", errors="coerce")
    return df


def download_index_bar1d(
    index_instrument: str,
    end_date: str,
    days: int,
    output_file: Path,
) -> Path:
    end_dt = pd.to_datetime(end_date)
    start_dt = end_dt - timedelta(days=days)
    start_date = start_dt.strftime("%Y-%m-%d")

    existing = _read_existing(output_file)
    if existing is not None and not existing.empty and "date" in existing.columns:
        latest = existing["date"].max()
        if pd.notna(latest) and latest >= end_dt:
            print(f"[OK] 指数数据已是最新（到 {latest.strftime('%Y-%m-%d')}），无需下载")
            return output_file
        if pd.notna(latest):
            start_date = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"[OK] 增量下载：{start_date} 至 {end_date}")

    dai = _import_bigquant_dai()

    sql = f"""
    SELECT *
    FROM cn_stock_index_bar1d
    WHERE instrument = '{index_instrument}'
      AND date >= '{start_date}' AND date <= '{end_date}'
    """
    df = dai.query(sql, full_db_scan=True).df()
    if df is None or df.empty:
        print(f"[WARN] 未获取到指数数据（{start_date} 至 {end_date}）")
        return output_file

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="ISO8601", errors="coerce")

    if existing is not None and not existing.empty:
        merged = pd.concat([existing, df], ignore_index=True)
        if "instrument" in merged.columns and "date" in merged.columns:
            merged = merged.drop_duplicates(subset=["instrument", "date"], keep="last")
        merged = merged.sort_values(["date"])
        added = len(merged) - len(existing)
        print(f"[OK] 合并完成：原有 {len(existing)} 条，本次下载 {len(df)} 条，实际增量 {added} 条")
        df = merged

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"[OK] 已保存指数数据: {output_file}（rows={len(df)}）")
    return output_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", dest="index_instrument", default="000001.SH", help="指数 instrument（默认 000001.SH 上证）")
    ap.add_argument("--end", dest="end_date", default=datetime.now().strftime("%Y-%m-%d"), help="结束日期 YYYY-MM-DD")
    ap.add_argument("--days", dest="days", type=int, default=180, help="下载/回看天数（默认 180）")
    ap.add_argument(
        "--out",
        dest="out",
        # 默认写到 AlphaSignal-CN/data/raw，避免在不同 cwd 下出现路径重复拼接
        default="data/raw/index_bar1d.csv",
        help="输出文件路径（相对路径会按 AlphaSignal-CN 工程根目录解析）",
    )
    args = ap.parse_args()

    # 将相对路径统一解析到 AlphaSignal-CN 工程根（scripts/ 的上一级）
    project_root = Path(__file__).resolve().parent.parent
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path

    download_index_bar1d(
        index_instrument=args.index_instrument,
        end_date=args.end_date,
        days=args.days,
        output_file=out_path,
    )


if __name__ == "__main__":
    main()

