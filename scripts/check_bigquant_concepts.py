"""
验证：某只股票在 BigQuant 的概念归属（概念名称/代码）

数据来源：cn_stock_index_concept_component
用法示例：
  python scripts/check_bigquant_concepts.py --symbol 300346.SZ
  python scripts/check_bigquant_concepts.py --symbol 300346 --days 60
  python scripts/check_bigquant_concepts.py --symbol 300346.SZ --date 2026-01-15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional


def _normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if "." in s:
        return s
    # 仅输入 6 位代码时，按常见规则补交易所后缀
    if len(s) == 6 and s.isdigit():
        if s.startswith(("60", "68")):
            return f"{s}.SH"
        if s.startswith(("92", "43", "8")):
            return f"{s}.BJ"
        return f"{s}.SZ"
    return s


def _parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


@dataclass(frozen=True)
class QueryWindow:
    start: str
    end: str


def _build_window(days: int, fixed_date: Optional[date]) -> QueryWindow:
    if fixed_date is not None:
        d = fixed_date.strftime("%Y-%m-%d")
        return QueryWindow(start=d, end=d)
    end = date.today()
    start = end - timedelta(days=days)
    return QueryWindow(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))


def main() -> int:
    parser = argparse.ArgumentParser(description="查询 BigQuant：个股概念归属（概念名称/代码）")
    parser.add_argument("--symbol", required=True, help="股票代码，如 300346 或 300346.SZ")
    parser.add_argument("--days", type=int, default=30, help="查询最近 N 天（默认 30）")
    parser.add_argument("--date", type=str, default=None, help="指定日期 YYYY-MM-DD（指定后忽略 --days）")
    args = parser.parse_args()

    symbol = _normalize_symbol(args.symbol)
    fixed_date = _parse_yyyy_mm_dd(args.date) if args.date else None
    window = _build_window(days=max(1, args.days), fixed_date=fixed_date)

    try:
        from bigquantdai import dai  # type: ignore
    except Exception as e:
        print("❌ 无法导入 bigquantdai。请确认你已在 venv 中安装并配置 BigQuant SDK。")
        print(f"   具体错误: {e}")
        return 2

    sql = (
        "SELECT date, instrument, name, member_code, member_name "
        "FROM cn_stock_index_concept_component "
        f"WHERE date >= '{window.start}' AND date <= '{window.end}' "
        f"AND member_code = '{symbol}' "
        "ORDER BY date DESC"
    )

    print("=== BigQuant 概念归属查询 ===")
    print(f"symbol: {symbol}")
    print(f"window: {window.start} ~ {window.end}")
    print(f"sql: {sql}")

    try:
        # BigQuant 某些表需要 filters 指定分区范围
        try:
            df = dai.query(sql, filters={"date": [window.start, window.end]}).df()
        except TypeError:
            df = dai.query(sql).df()
    except Exception as e:
        print("❌ 查询失败（可能是 token/权限/filters 分区范围问题）")
        print(f"   错误: {e}")
        return 3

    if df is None or df.empty:
        print("⚠️ 未查询到记录（可能该日期范围内无数据，或代码格式不匹配）")
        return 0

    latest = df["date"].max()
    latest_df = df[df["date"] == latest][["instrument", "name"]].drop_duplicates()
    latest_df = latest_df.sort_values(["name", "instrument"])

    print(f"\nlatest_date: {latest}")
    print(f"concept_count: {len(latest_df)}")
    print("\n--- concepts ---")
    for _, r in latest_df.iterrows():
        print(f"{r['instrument']}\t{r['name']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

