"""
将 AlphaSignal（predict_stock.py --batch）输出的 predictions_trigger_YYYYMMDD.json
转换为 BigQuant 回测可用的“每日 TOP3 信号表”（CSV）。

核心约定（避免未来函数）：
- predictions_trigger_YYYYMMDD.json 里的 trigger_date 通常是 T（日内异动/涨停日）。
- 这些结果在 T 日收盘后生成，更适合作为 T+1 的买入信号。
- 因此本脚本默认将 buy_day = next_trading_day(trigger_date)。

输出 CSV schema 见：
  docs/BigQuant_短线接力TOP3_signals_schema.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SignalRow:
    date: str
    instrument: str
    score: float
    rank: int
    weight: float
    source_date: str
    model: str


def _try_load_bigquant_trading_days(year: int) -> set[pd.Timestamp] | None:
    """
    可选：用 BigQuant all_trading_days 获取交易日（更准确，含节假日）。
    若未安装/未登录 BigQuant SDK，则返回 None，调用方会回退到工作日规则。
    """
    try:
        from bigquantdai import dai  # type: ignore
    except Exception:
        try:
            from bigquant.api import dai  # type: ignore
        except Exception:
            return None

    start = f"{year}-01-01"
    end = f"{year}-12-31"
    sql = f"""
    SELECT date
    FROM all_trading_days
    WHERE date >= '{start}' AND date <= '{end}'
    """
    df = dai.query(sql, full_db_scan=True).df()
    if df is None or df.empty:
        return None
    col = "date" if "date" in df.columns else df.columns[0]
    ds = pd.to_datetime(df[col], errors="coerce").dropna()
    return set(ds.dt.normalize().tolist())


def _next_trading_day(d: str) -> str:
    """
    计算下一个交易日（优先 BigQuant all_trading_days；失败则回退到周一~周五）。
    """
    dt = pd.to_datetime(d).normalize()

    days = _try_load_bigquant_trading_days(dt.year) or _try_load_bigquant_trading_days(dt.year + 1)
    if days:
        cur = dt + pd.Timedelta(days=1)
        for _ in range(370):
            if cur in days:
                return cur.strftime("%Y-%m-%d")
            cur += pd.Timedelta(days=1)
        # 极端兜底：回退工作日

    cur = dt + pd.Timedelta(days=1)
    for _ in range(370):
        if cur.weekday() < 5:
            return cur.strftime("%Y-%m-%d")
        cur += pd.Timedelta(days=1)
    return (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _iter_prediction_files(results_dir: Path) -> Iterable[Path]:
    yield from sorted(results_dir.glob("predictions_trigger_*.json"))


def _load_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # 兼容：若是 dict 包一层
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]
    raise ValueError(f"Unsupported predictions JSON format: {path}")


def build_signals(
    results_dir: Path,
    topk: int,
    min_score: float | None,
    model_name: str,
) -> pd.DataFrame:
    rows: list[SignalRow] = []

    for fp in _iter_prediction_files(results_dir):
        preds = _load_predictions(fp)
        if not preds:
            continue

        # trigger_date 来自每条结果；理论上同文件一致，但这里做一次稳健处理
        trigger_date = None
        # 取这个文件里 score 最大的那批（按 final_prob）
        cleaned: list[tuple[str, float]] = []
        for r in preds:
            symbol = str(r.get("symbol", "") or "")
            td = str(r.get("trigger_date", "") or "")
            score = r.get("final_prob", None)
            if not symbol or "." not in symbol:
                continue
            try:
                score_f = float(score)
            except Exception:
                continue
            if min_score is not None and score_f < min_score:
                continue
            cleaned.append((symbol, score_f))
            if td:
                trigger_date = td

        if not cleaned or not trigger_date:
            continue

        buy_day = _next_trading_day(trigger_date)

        # TOPK by score desc
        cleaned.sort(key=lambda x: x[1], reverse=True)
        picked = cleaned[:topk]
        if not picked:
            continue

        weight = 1.0 / len(picked)
        for i, (instrument, score_f) in enumerate(picked, start=1):
            rows.append(
                SignalRow(
                    date=buy_day,
                    instrument=instrument,
                    score=score_f,
                    rank=i,
                    weight=weight,
                    source_date=trigger_date,
                    model=model_name,
                )
            )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return df

    # 同一 buy_day 内，去重并保留最高分
    df = df.sort_values(["date", "score"], ascending=[True, False])
    df = df.drop_duplicates(subset=["date", "instrument"], keep="first")

    # 重新在每日内排 rank、归一 weight（确保一定是 TOPK）
    df["rank"] = (
        df.groupby("date")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    df = df[df["rank"] <= topk].copy()
    df["weight"] = df.groupby("date")["rank"].transform(lambda s: 1.0 / len(s))

    # 排序输出
    df = df.sort_values(["date", "rank"]).reset_index(drop=True)
    return df


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default=str(repo_root / "results"),
        help="AlphaSignal 预测结果目录（包含 predictions_trigger_*.json）",
    )
    ap.add_argument("--topk", type=int, default=3, help="每个交易日取 TOPK（默认 3）")
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="可选：过滤最低分（默认 0.5；设为 -1 表示不过滤）",
    )
    ap.add_argument(
        "--model",
        default="alphasignal_relay_v1",
        help="写入输出的 model 字段",
    )
    ap.add_argument(
        "--out",
        default=str(repo_root / "data" / "raw" / "bigquant_signals_top1.csv"),
        help="输出 CSV 路径",
    )
    args = ap.parse_args()

    min_score = None if args.min_score is None or args.min_score < 0 else float(args.min_score)

    results_dir = Path(args.results_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_signals(
        results_dir=results_dir,
        topk=int(args.topk),
        min_score=min_score,
        model_name=str(args.model),
    )
    if df.empty:
        raise SystemExit(f"No signals generated from: {results_dir}")

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] signals saved: {out_path} (rows={len(df)}, days={df['date'].nunique()})")


if __name__ == "__main__":
    main()

