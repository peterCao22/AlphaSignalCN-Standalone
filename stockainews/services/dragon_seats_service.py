"""
龙虎榜席位明细（同花顺 market/longhu）SQLite 入库服务

存储目标：
- 每个 (trade_date, window_days, category, stock_code) 一条 summary
- 每个 summary 对应两组明细：buy_top5 / sell_top5
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlite3

from stockainews.core.logger import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class DragonSeatKey:
    trade_date: str
    window_days: int
    category: str
    stock_code: str


class DragonSeatsService:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = (
                project_root
                / "TradingAgents-chinese-market"
                / "AlphaSignal-CN"
                / "data"
                / "dragon_seats.db"
            )

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"龙虎榜席位DB初始化完成: {self.db_path}")

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dragon_seat_summary (
                trade_date TEXT NOT NULL,
                window_days INTEGER NOT NULL,
                category TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                stock_name TEXT,
                reason TEXT,
                total_turnover_yuan REAL,
                total_buy_yuan REAL,
                total_sell_yuan REAL,
                total_net_yuan REAL,
                crawl_date TEXT,
                PRIMARY KEY (trade_date, window_days, category, stock_code)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dragon_seat_detail (
                trade_date TEXT NOT NULL,
                window_days INTEGER NOT NULL,
                category TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                side TEXT NOT NULL,            -- buy / sell
                rank INTEGER NOT NULL,         -- 1..5
                seat_name TEXT NOT NULL,
                seat_tag TEXT,
                buy_yuan REAL,
                sell_yuan REAL,
                net_yuan REAL,
                PRIMARY KEY (trade_date, window_days, category, stock_code, side, rank)
            )
            """
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dragon_seat_detail_seat ON dragon_seat_detail(seat_name)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dragon_seat_detail_date ON dragon_seat_detail(trade_date)"
        )

        conn.commit()
        conn.close()

    def save_records(self, records: List[Dict[str, Any]], crawl_date: Optional[date] = None) -> None:
        if not records:
            logger.warning("席位明细 records 为空，跳过入库")
            return

        crawl_date_str = (crawl_date or date.today()).strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        try:
            for r in records:
                key = DragonSeatKey(
                    trade_date=str(r.get("trade_date", ""))[:10],
                    window_days=int(r.get("window_days", 1) or 1),
                    category=str(r.get("category", "")),
                    stock_code=str(r.get("stock_code", "")),
                )

                # upsert summary
                cur.execute(
                    """
                    INSERT OR REPLACE INTO dragon_seat_summary (
                        trade_date, window_days, category, stock_code,
                        stock_name, reason,
                        total_turnover_yuan, total_buy_yuan, total_sell_yuan, total_net_yuan,
                        crawl_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key.trade_date,
                        key.window_days,
                        key.category,
                        key.stock_code,
                        r.get("stock_name", ""),
                        r.get("reason", ""),
                        r.get("total_turnover_yuan", None),
                        r.get("total_buy_yuan", None),
                        r.get("total_sell_yuan", None),
                        r.get("total_net_yuan", None),
                        crawl_date_str,
                    ),
                )

                # delete old details for this key to avoid stale rows
                cur.execute(
                    """
                    DELETE FROM dragon_seat_detail
                    WHERE trade_date=? AND window_days=? AND category=? AND stock_code=?
                    """,
                    (key.trade_date, key.window_days, key.category, key.stock_code),
                )

                for side, lst in (("buy", r.get("buy_top5", []) or []), ("sell", r.get("sell_top5", []) or [])):
                    for item in lst:
                        cur.execute(
                            """
                            INSERT OR REPLACE INTO dragon_seat_detail (
                                trade_date, window_days, category, stock_code,
                                side, rank, seat_name, seat_tag,
                                buy_yuan, sell_yuan, net_yuan
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                key.trade_date,
                                key.window_days,
                                key.category,
                                key.stock_code,
                                side,
                                int(item.get("rank", 0) or 0),
                                str(item.get("seat_name", "")),
                                item.get("seat_tag", None),
                                item.get("buy_yuan", None),
                                item.get("sell_yuan", None),
                                item.get("net_yuan", None),
                            ),
                        )

            conn.commit()
            logger.info(f"[OK] 龙虎榜席位明细已入库: {len(records)} 条 summary（含明细）")
        except Exception as e:
            conn.rollback()
            logger.error(f"入库失败: {e}", exc_info=True)
            raise
        finally:
            conn.close()

