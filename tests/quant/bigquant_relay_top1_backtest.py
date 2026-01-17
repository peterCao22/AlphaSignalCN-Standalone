"""
BigQuant 一年回测：短线接力模型（资金有限：每天最多买 1 只，持有 N 个交易日滚动）

使用方式（在 BigQuant AIStudio / notebook 中运行）：
1) 上传信号 CSV（见 docs/BigQuant_短线接力TOP3_signals_schema.md）
2) 修改 SIGNALS_CSV / START_DATE / END_DATE
3) 运行本脚本（或把核心函数复制到你的策略里）

交易规则（简化版，A-日线、严格“今天信号→延后买→持有N日→到期开盘卖”）：
- 信号日：D（日线 K 已收盘时收到信号）
- 买入：按 signals CSV 的 `date` 作为“下单日”（当日收盘后下单），在下一交易日开盘撮合；
        等价于：如果 `date` 原本是 D+1，则现在实际买入是在 D+2 开盘
- 卖出：买入成交日起算持有 N 个交易日；在“到期日的前一交易日收盘后”下单，到期日开盘按 open 价点撮合清仓卖出

实现说明（重要）：
- 由于你当前账号无 cn_stock_bar1m（分钟线）权限，因此回测改为日线撮合。
- BigTrader 日线回测时序：handle_data 在“当日收盘”触发；当日下的单会在“下一交易日”撮合。
- 本脚本把买/卖都放在 handle_data，以保证撮合时点与持有期严格一致。
  文档参考：https://bigquant.com/wiki/doc/3gG2rg4jBd
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd


# ======== 你需要改的配置 ========
SIGNALS_CSV = "/home/aiuser/work/bigquant_signals_top1.csv"  # AIStudio 上传后路径示例
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
CAPITAL_BASE = 5_0000
EXPORT_RAW_PERF_CSV = "/home/aiuser/work/raw_perf_relay_top1.csv"  # 可改为空字符串表示不导出

# 每个信号日最多买入数量（资金有限：1）
TOPK = 1
MODEL_NAME = "alphasignal_relay_v1"

# === 持有期（按交易日计）===
# - 例如：buy_exec_day=2025-01-03，HOLD_TRADING_DAYS=5，则目标卖出成交日为 2025-01-10（开盘卖）
HOLD_TRADING_DAYS = 5

# === 卖出规则：均线破位止损（规则H，方案A：加二次确认，减少误杀）===
# - 回测为日线撮合：触发在“收盘后”判断，因此只能做到“次日开盘卖出”
# - 方案A：仅当“收盘跌破 MA5”且满足二次确认之一才卖：
#   1) 同时跌破 MA10（更偏“趋势走坏”）
#   2) 或当日跌幅 <= STOP_DAILY_RETURN_THRESHOLD（更偏“大阴线”）
ENABLE_STOP_ON_CLOSE_BELOW_MA5 = True
MA5_WINDOW = 5
MA10_WINDOW = 10
STOP_REQUIRE_BELOW_MA10 = True
STOP_ALLOW_BIG_DOWN_DAY = False
STOP_DAILY_RETURN_THRESHOLD = -0.05  # -5%

# === 智能缩短回测区间（避免“信号只有几天，但回测空跑到年底”）===
# - 会根据 signals CSV 中最后一个信号日 D，自动把 end_date 收敛到 D + padding_days（自然日）
# - 这里采用“方案C”：padding_days 由（买入延后 + 持有期）自动估算，并再加节假日缓冲
AUTO_END_DATE = True
# 额外节假日/不交易日缓冲（自然日）。春节/国庆等长假前后建议调大。
AUTO_END_DATE_EXTRA_BUFFER_DAYS = 20

# === 每日新开仓位（滚动持仓用）===
# - 目标：持有期为 N 个交易日时，每天只新开 1/N 的仓位；当日 TOP3 在这 1/N 内等权分配
# - 这样可以在最多 N 天叠加到接近满仓，避免“持有期变长导致后续批次 NoCashAvail”
DAILY_NEW_POSITION_FRACTION = 1.0 / float(HOLD_TRADING_DAYS)

# === 方案A：signals CSV 口径（推荐）===
# - date: buy_day（信号日 D 的下一交易日，用于审计）
# - source_date: 信号日 D（策略以此驱动下单；缺失则回退使用 date）
USE_SOURCE_DATE_AS_SIGNAL_DAY = True

# === 买入延后（更贴近“等一天确认”）===
# - 本脚本在日线撮合下：当天收盘后下单 -> 下一交易日开盘成交
# - 若设为 1，则“按 CSV date 下单”，实际买入会比 CSV date 再晚 1 个交易日开盘
# - 若设为 2，则“在 CSV date 的下一交易日收盘后下单”，实际买入会比 CSV date 再晚 2 个交易日开盘
BUY_DELAY_FROM_CSV_DATE_TRADING_DAYS = 2


def _calc_auto_end_date_padding_days() -> int:
    """
    方案C：用（交易日需求 * 7/5）换算到自然日，再加额外缓冲，避免回测区间截断尾部卖出。

    需要覆盖的“交易日需求”大致包含：
    - BUY_DELAY_FROM_CSV_DATE_TRADING_DAYS（下单到成交的延后）
    - HOLD_TRADING_DAYS（持有期，到期开盘卖）
    - +1（撮合/顺延的保守余量）
    """
    delay_td = max(int(BUY_DELAY_FROM_CSV_DATE_TRADING_DAYS), 0)
    hold_td = max(int(HOLD_TRADING_DAYS), 0)
    need_trading_days = delay_td + hold_td + 1

    # 将交易日近似换算为自然日（A股大致 5/7 交易日密度）
    base_natural_days = int(math.ceil((7.0 / 5.0) * float(need_trading_days)))
    extra = max(int(AUTO_END_DATE_EXTRA_BUFFER_DAYS), 0)
    return base_natural_days + extra


def _ymd_to_yyyymmdd(s: str) -> str:
    s = str(s or "").strip()
    if len(s) == 8 and s.isdigit():
        return s
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        y, m, d = s[:4], s[5:7], s[8:10]
        if y.isdigit() and m.isdigit() and d.isdigit():
            return f"{y}{m}{d}"
    return s.replace("-", "")


def _yyyymmdd_to_ymd(s: str) -> str:
    s = str(s or "").strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def _get_trading_day_yyyymmdd(context, data) -> str:
    # BigTrader 常见返回：'YYYYmmdd'
    try:
        td = str(getattr(context, "get_trading_day")())
        td = td.strip()
        if td:
            return _ymd_to_yyyymmdd(td)
    except Exception:
        pass
    # 兜底：用 current_dt 推出交易日
    return data.current_dt.strftime("%Y%m%d")


def _get_next_trading_day_yyyymmdd(context, cur_trading_day_yyyymmdd: str) -> str:
    cur = _ymd_to_yyyymmdd(cur_trading_day_yyyymmdd)

    # BigQuant BigTrader 常见接口：context.get_next_trading_day(cur_trading_day='YYYYmmdd') -> 'YYYYmmdd'
    fn = getattr(context, "get_next_trading_day", None)
    if callable(fn):
        try:
            nxt = fn(cur_trading_day=cur)
            return _ymd_to_yyyymmdd(str(nxt))
        except TypeError:
            nxt = fn(cur)
            return _ymd_to_yyyymmdd(str(nxt))

    # 兜底：如果环境提供 trading_calendar
    cal = getattr(context, "trading_calendar", None)
    if cal is not None:
        fn2 = getattr(cal, "get_next_trading_day", None)
        if callable(fn2):
            return _ymd_to_yyyymmdd(str(fn2(cur)))

    raise RuntimeError("cannot get next trading day: missing context.get_next_trading_day")


def _add_trading_days_yyyymmdd(context, start_trading_day_yyyymmdd: str, n: int) -> str:
    td = _ymd_to_yyyymmdd(start_trading_day_yyyymmdd)
    for _ in range(int(n)):
        td = _get_next_trading_day_yyyymmdd(context, td)
    return td


def _safe_get_last_price_from_position(pos) -> float | None:
    """
    raw_perf 的 positions 字段里通常有 last_price（当日收盘价），这里做宽松兼容（dict / object）。
    """
    try:
        if isinstance(pos, dict):
            v = pos.get("last_price", None)
            return float(v) if v is not None else None
        v = getattr(pos, "last_price", None)
        return float(v) if v is not None else None
    except Exception:
        return None


def _get_recent_closes(data, instrument: str, n: int) -> list[float] | None:
    """
    尝试从 BigTrader 的 data 对象获取最近 n 根日线 close，用于计算 MA5。
    不同环境接口可能略有差异，这里多路兜底；取不到则返回 None（不触发 MA 止损）。
    """
    n = int(n)
    if n <= 0:
        return None

    fn = getattr(data, "history", None)
    if callable(fn):
        # BigTrader 回测环境中常见签名：
        # PyBarDatas_history(string symbol, string field, int bar_count, string frequency)
        try:
            out = fn(instrument, "close", n, "1d")
        except Exception:
            out = None

        if out is not None:
            try:
                # list / tuple
                if isinstance(out, (list, tuple)):
                    vals = [float(x) for x in out if x is not None]
                # dict
                elif isinstance(out, dict):
                    v = out.get("close") or out.get("Close") or out.get("CLOSE")
                    if isinstance(v, (list, tuple)):
                        vals = [float(x) for x in v if x is not None]
                    else:
                        vals = [float(v)] if v is not None else []
                else:
                    # pandas-like / array-like
                    v = None
                    try:
                        v = out["close"]
                    except Exception:
                        pass
                    if v is not None:
                        vals = [float(x) for x in list(v) if x is not None]
                    else:
                        vals = [float(x) for x in list(out) if x is not None]
            except Exception:
                vals = []

            if len(vals) >= n:
                return vals[-n:]

    fn2 = getattr(data, "get_price", None)
    if callable(fn2):
        candidates2 = [
            lambda: fn2(instrument, count=n, fields=["close"]),
            lambda: fn2(instrument, count=n, fields="close"),
        ]
        for getter in candidates2:
            try:
                out = getter()
                if out is None:
                    continue
                v = None
                try:
                    v = out["close"]
                except Exception:
                    pass
                if v is None:
                    continue
                vals = [float(x) for x in list(v) if x is not None]
                if len(vals) >= n:
                    return vals[-n:]
            except Exception:
                continue

    return None


def _compute_auto_end_date(signals_csv: str, padding_days: int) -> str | None:
    try:
        df = pd.read_csv(signals_csv, dtype={"date": str, "source_date": str})
    except Exception:
        return None
    if df is None or df.empty or "date" not in df.columns:
        return None
    if USE_SOURCE_DATE_AS_SIGNAL_DAY and "source_date" in df.columns:
        s = df["source_date"].astype(str).str.slice(0, 10)
        if s.isna().all() or (s == "").all():
            s = df["date"].astype(str).str.slice(0, 10)
    else:
        s = df["date"].astype(str).str.slice(0, 10)
    s = s[s.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    if s.empty:
        return None
    last_d = s.max()
    try:
        end_dt = pd.to_datetime(last_d) + pd.Timedelta(days=int(padding_days))
        return end_dt.strftime("%Y-%m-%d")
    except Exception:
        return None


@dataclass(frozen=True)
class DaySignal:
    instruments: list[str]
    weights: list[float]
    scores: list[float]
    source_dates: list[str]
    buy_days: list[str]


def _load_signals(path: str) -> dict[str, DaySignal]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"signals csv not found: {path}")

    df = pd.read_csv(p, dtype={"date": str, "instrument": str, "source_date": str})
    if df.empty:
        raise ValueError("signals csv is empty")

    # 清洗代码字段：去空格、去尾部点（避免出现 603068.SH. 这种导致数据/撮合异常）
    df["instrument"] = df["instrument"].astype(str).str.strip().str.rstrip(".")
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    if "source_date" in df.columns:
        df["source_date"] = df["source_date"].astype(str).str.slice(0, 10)

    required = {"date", "instrument"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"signals csv missing columns: {sorted(list(missing))}")

    if "score" not in df.columns:
        df["score"] = 0.0
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

    # 下单日（base_order_day）：用于决定“哪天收盘后下单”
    # - 这里先按 CSV 的 date 作为 base_order_day
    # - 实际下单日会在 initialize() 内根据 BUY_DELAY_FROM_CSV_DATE_TRADING_DAYS 做“交易日偏移”
    # - 注意：CSV 的 source_date 仍保留用于审计/分析，但不再驱动下单
    df["_order_day"] = df["date"]

    # 每天只取 TOPK，按 score 降序
    df = df.sort_values(["_order_day", "score"], ascending=[True, False])
    df["rank"] = df.groupby("_order_day")["score"].rank(method="first", ascending=False)
    df = df[df["rank"] <= TOPK].copy()

    # 权重：等权（每个 buy_day 可能不足 TOPK，则按实际数量等权）
    df["weight"] = df.groupby("_order_day")["instrument"].transform(lambda s: 1.0 / len(s))

    if "source_date" not in df.columns:
        df["source_date"] = ""

    out: dict[str, DaySignal] = {}
    for d, g in df.groupby("_order_day", sort=True):
        out[str(d)] = DaySignal(
            instruments=g["instrument"].astype(str).tolist(),
            weights=g["weight"].astype(float).tolist(),
            scores=g["score"].astype(float).tolist(),
            source_dates=g["source_date"].astype(str).tolist(),
            buy_days=g["date"].astype(str).tolist(),
        )
    return out


def initialize(context):
    from bigtrader.finance.commission import PerOrder

    context.set_commission(PerOrder(buy_cost=0.0005, sell_cost=0.0013, min_cost=5))

    # 读取信号表（全量一次性读入）
    signals_by_day = _load_signals(SIGNALS_CSV)

    # 将 CSV 的 date（base_order_day）按交易日偏移到“实际下单日”
    # - BUY_DELAY_FROM_CSV_DATE_TRADING_DAYS = 1 => shift 0（按 CSV date 当日收盘下单，次日开盘买）
    # - BUY_DELAY_FROM_CSV_DATE_TRADING_DAYS = 2 => shift 1（延后一天收盘下单，次日开盘买）
    shift_n = max(int(BUY_DELAY_FROM_CSV_DATE_TRADING_DAYS) - 1, 0)
    if shift_n > 0:
        shifted: dict[str, DaySignal] = {}
        for base_order_day, sig in signals_by_day.items():
            try:
                base_td = _ymd_to_yyyymmdd(base_order_day)
                shifted_td = _add_trading_days_yyyymmdd(context, base_td, shift_n)
                shifted_day = _yyyymmdd_to_ymd(shifted_td)
            except Exception:
                # 极端兜底：若无法计算交易日，则不偏移
                shifted_day = str(base_order_day)

            # 正常情况下不同 base_day 偏移后不会冲突；冲突时保留更高分的一组
            if shifted_day in shifted:
                prev = shifted[shifted_day]
                prev_score = float(prev.scores[0]) if prev.scores else -1.0
                cur_score = float(sig.scores[0]) if sig.scores else -1.0
                if cur_score > prev_score:
                    shifted[shifted_day] = sig
            else:
                shifted[shifted_day] = sig
        signals_by_day = shifted

    context.user_store["signals_by_day"] = signals_by_day
    context.user_store["model"] = str(MODEL_NAME)

    # 记录每个持仓的 buy_day（用于判断卖出日）
    # 记录每个持仓的“实际买入成交日”（YYYY-MM-DD），用于次日卖出
    context.user_store["buy_exec_day_by_instrument"] = {}


def before_trading_start(context, data):
    # 日线回测：不在盘前下单，避免撮合时点错位
    return


def handle_data(context, data):
    """
    日线回测（关键）：handle_data 表示“当日已经收盘”，此时下单会在下一交易日撮合。

    执行顺序（都在今天收盘后下单，下一交易日开盘撮合）：
    1) 若“下一交易日”是某持仓的到期卖出日，则在今天收盘后下达清仓卖出（实现：到期日开盘卖）
    2) 读取今天的 signals（信号日 D），下达明日开盘买入 TOP3
    """
    today = data.current_dt.strftime("%Y-%m-%d")
    signals_by_day: dict[str, DaySignal] = context.user_store.get("signals_by_day", {})
    buy_exec_day_by_instrument: dict[str, str] = context.user_store.get("buy_exec_day_by_instrument", {})

    today_td = _get_trading_day_yyyymmdd(context, data)
    next_td = _get_next_trading_day_yyyymmdd(context, today_td)

    # ========= 1) 卖出：到期卖出 + 收盘跌破 MA5 强制卖出（规则H） =========
    positions = context.get_positions()
    for instrument in list(positions.keys()):
        buy_exec_td = buy_exec_day_by_instrument.get(instrument)
        if not buy_exec_td:
            continue

        # (A) 到期卖出：目标卖出成交日（开盘卖）= buy_exec_day + HOLD_TRADING_DAYS 个交易日
        try:
            target_sell_td = _add_trading_days_yyyymmdd(context, buy_exec_td, HOLD_TRADING_DAYS)
        except Exception as e:
            context.record_log("WARN", f"[SELL-CALC-FAIL] {instrument} buy_exec_td={buy_exec_td} err={e}", "")
            continue

        expire_hit = str(next_td) == str(target_sell_td)

        # (B) 均线破位止损（规则H / 方案A）
        ma_hit = False
        close_px = None
        ma5 = None
        ma10 = None
        daily_ret = None
        if ENABLE_STOP_ON_CLOSE_BELOW_MA5:
            need_n = max(int(MA5_WINDOW), int(MA10_WINDOW), 2)
            closes = _get_recent_closes(data, instrument, need_n)
            if closes and len(closes) >= max(int(MA5_WINDOW), 2):
                # 优先用 history 的最后一个 close，当作“收盘价”
                close_px = float(closes[-1])

                # MA5
                if len(closes) >= int(MA5_WINDOW) and int(MA5_WINDOW) > 1:
                    ma5 = sum(closes[-int(MA5_WINDOW) :]) / float(int(MA5_WINDOW))

                # MA10（可选）
                if len(closes) >= int(MA10_WINDOW) and int(MA10_WINDOW) > 1:
                    ma10 = sum(closes[-int(MA10_WINDOW) :]) / float(int(MA10_WINDOW))

                # 当日涨跌幅（用 close-to-close 近似）
                if len(closes) >= 2 and float(closes[-2]) != 0.0:
                    daily_ret = float(closes[-1]) / float(closes[-2]) - 1.0
            else:
                # 兜底：若取不到 close 序列，则不触发 MA 止损
                close_px = _safe_get_last_price_from_position(positions.get(instrument))

            below_ma5 = (
                close_px is not None
                and ma5 is not None
                and float(close_px) < float(ma5)
            )
            below_ma10 = (
                close_px is not None
                and ma10 is not None
                and float(close_px) < float(ma10)
            )
            big_down = (
                daily_ret is not None
                and float(daily_ret) <= float(STOP_DAILY_RETURN_THRESHOLD)
            )

            if below_ma5:
                confirm = False
                if STOP_REQUIRE_BELOW_MA10 and below_ma10:
                    confirm = True
                if STOP_ALLOW_BIG_DOWN_DAY and big_down:
                    confirm = True
                ma_hit = confirm

        # 两个条件都未触发，则不卖
        if not expire_hit and not ma_hit:
            continue

        if not data.can_trade(instrument):
            context.record_log("WARN", f"[SELL-DELAY] {instrument} cannot trade on {today}, postpone", "")
            continue
        ret = context.order_target_percent(instrument, 0.0)
        if ret != 0:
            msg = context.get_error_msg(ret)
            context.record_log("WARN", f"[SELL-FAIL] {instrument} ret={ret} msg={msg}", "")
        else:
            if ma_hit:
                context.record_log(
                    "INFO",
                    f"[SELL-MA5A-NEXT-OPEN] {instrument} close={close_px} ma5={ma5} ma10={ma10} daily_ret={daily_ret} (buy_exec_day={_yyyymmdd_to_ymd(buy_exec_td)})",
                    "",
                )
            else:
                context.record_log(
                    "INFO",
                    f"[SELL-NEXT-OPEN] {instrument} buy_exec_day={_yyyymmdd_to_ymd(buy_exec_td)} target_sell_day={_yyyymmdd_to_ymd(target_sell_td)}",
                    "",
                )

    # ========= 2) 买入：读取“今天下单日（CSV date）”的 TOP1，下单下一交易日开盘买入 =========
    sig = signals_by_day.get(today)
    if not sig:
        return

    # 滚动持仓：每天只新开 1/N 的仓位（N=HOLD_TRADING_DAYS），并在当日 TOP3 内等权分配。
    # 说明：为避免同一标的“分批加仓/分批到期卖出”的复杂性，这里若已持有则跳过当日新买入。
    daily_frac = float(DAILY_NEW_POSITION_FRACTION) if float(HOLD_TRADING_DAYS) > 0 else 0.0
    holding_instruments = set(positions.keys())

    for instrument, w, sc, src, buy_day in zip(
        sig.instruments, sig.weights, sig.scores, sig.source_dates, sig.buy_days
    ):
        if not data.can_trade(instrument):
            context.record_log("WARN", f"[BUY-SKIP] {instrument} cannot trade on {today}", "")
            continue

        if instrument in holding_instruments:
            context.record_log("INFO", f"[BUY-SKIP-HOLDING] {instrument} already holding on {today}", "")
            continue

        # 注意：严格口径下即便同一只股票连续入选，也应“先卖后买”。
        # 但日线撮合在同一价点（次日开盘）发生，且 bigtrader 可能会净额撮合；
        # 这里不强制拆分，按“买入信号 -> 下单买入”处理。
        target_w = float(w) * daily_frac
        ret = context.order_target_percent(instrument, target_w)
        if ret != 0:
            msg = context.get_error_msg(ret)
            context.record_log("WARN", f"[BUY-FAIL] {instrument} ret={ret} msg={msg}", "")
            continue
        context.record_log(
            "INFO",
            f"[BUY-NEXT-OPEN] order_day={today} exec_day={_yyyymmdd_to_ymd(next_td)} csv_date={buy_day} {instrument} w={target_w:.3f} score={float(sc):.3f} source_date={src}",
            "",
        )


def handle_trade(context, trade):
    """
    以真实成交回报为准，记录“买入成交日”（交易日），用于“持有 N 个交易日后卖出”的严格持有期控制。
    """
    try:
        from bigtrader.constant import Direction  # type: ignore

        buy_flag = str(Direction.BUY)
        sell_flag = str(Direction.SELL)
    except Exception:
        buy_flag = "1"
        sell_flag = "2"

    instrument = getattr(trade, "instrument", None) or getattr(trade, "symbol", None)
    if not instrument:
        return

    # trading_day 期望为 'YYYYmmdd' 或 'YYYY-MM-DD'
    td = str(getattr(trade, "trading_day", "") or getattr(trade, "trade_date", "") or "")
    td = _ymd_to_yyyymmdd(td)

    direction = str(getattr(trade, "direction", ""))
    buy_exec_day_by_instrument: dict[str, str] = context.user_store.get("buy_exec_day_by_instrument", {})

    if direction == buy_flag or direction == "1":
        if td:
            buy_exec_day_by_instrument[str(instrument)] = td
    elif direction == sell_flag or direction == "2":
        buy_exec_day_by_instrument.pop(str(instrument), None)

    context.user_store["buy_exec_day_by_instrument"] = buy_exec_day_by_instrument


def after_trading(context, data):
    # 可在这里记录每天持仓/信号等，用于 raw_perf 分析
    return


def run_backtest():
    from bigquant import bigtrader

    end_date = END_DATE
    if AUTO_END_DATE:
        padding_days = _calc_auto_end_date_padding_days()
        auto_end = _compute_auto_end_date(SIGNALS_CSV, padding_days)
        if auto_end:
            # 若用户 END_DATE 更早，则以用户配置为准；否则用自动收敛后的日期
            try:
                end_date = min(str(END_DATE), str(auto_end))
            except Exception:
                end_date = str(auto_end)

    performance = bigtrader.run(
        market=bigtrader.Market.CN_STOCK,
        frequency=bigtrader.Frequency.DAILY,
        start_date=START_DATE,
        end_date=end_date,
        capital_base=CAPITAL_BASE,
        initialize=initialize,
        before_trading_start=before_trading_start,
        handle_data=handle_data,
        handle_trade=handle_trade,
        after_trading=after_trading,
        # 日线撮合：今天收盘下单 -> 下一交易日撮合；这里用 open 对齐“次日开盘买/卖”口径
        order_price_field_buy="open",
        order_price_field_sell="open",
        volume_limit=1,
        benchmark="000300.SH",
    )
    performance.render()

    # 可选：导出回测明细（raw_perf）
    try:
        out = str(EXPORT_RAW_PERF_CSV or "").strip()
        if out:
            performance.raw_perf.to_csv(out, index=False, encoding="utf-8")  # type: ignore
            print(f"[OK] raw_perf exported: {out} (rows={len(performance.raw_perf)})")  # type: ignore
    except Exception as e:
        print(f"[WARN] export raw_perf failed: {e}")

    return performance


if __name__ == "__main__":
    run_backtest()

