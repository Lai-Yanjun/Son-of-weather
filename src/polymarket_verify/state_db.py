from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class LedgerPosition:
    token_id: str
    condition_id: str
    outcome_index: int
    shares: float
    avg_price: float


@dataclass(frozen=True)
class LedgerState:
    cash_usdc: float
    realized_pnl_usdc: float


@dataclass(frozen=True)
class MarketMapping:
    condition_id: str
    outcome_index: int
    token_id: str
    question: str
    minimum_order_size: float
    tick_size: float
    neg_risk: bool


class StateDB:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path))
        c.row_factory = sqlite3.Row
        return c

    def _init(self) -> None:
        with self._connect() as c:
            c.execute("PRAGMA journal_mode=WAL;")
            c.execute("PRAGMA synchronous=NORMAL;")

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_trades (
                  k TEXT PRIMARY KEY,
                  ts INTEGER NOT NULL
                );
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS market_map (
                  condition_id TEXT NOT NULL,
                  outcome_index INTEGER NOT NULL,
                  token_id TEXT NOT NULL,
                  question TEXT NOT NULL,
                  minimum_order_size REAL NOT NULL,
                  tick_size REAL NOT NULL,
                  neg_risk INTEGER NOT NULL,
                  updated_at INTEGER NOT NULL,
                  PRIMARY KEY (condition_id, outcome_index)
                );
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_state (
                  id INTEGER PRIMARY KEY CHECK (id = 1),
                  cash_usdc REAL NOT NULL,
                  realized_pnl_usdc REAL NOT NULL,
                  updated_at INTEGER NOT NULL
                );
                """
            )
            c.execute(
                """
                INSERT OR IGNORE INTO ledger_state (id, cash_usdc, realized_pnl_usdc, updated_at)
                VALUES (1, 0.0, 0.0, 0);
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_positions (
                  token_id TEXT PRIMARY KEY,
                  condition_id TEXT NOT NULL,
                  outcome_index INTEGER NOT NULL,
                  shares REAL NOT NULL,
                  avg_price REAL NOT NULL,
                  updated_at INTEGER NOT NULL
                );
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                  ts INTEGER PRIMARY KEY,
                  cash_usdc REAL NOT NULL,
                  equity_usdc REAL NOT NULL,
                  total_cost_exposure_usdc REAL NOT NULL,
                  unrealized_pnl_usdc REAL NOT NULL
                );
                """
            )

    def get_kv(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as c:
            row = c.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def set_kv(self, key: str, value: str) -> None:
        with self._connect() as c:
            c.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def add_seen(self, *, k: str, ts: int) -> None:
        with self._connect() as c:
            c.execute("INSERT OR REPLACE INTO seen_trades(k, ts) VALUES (?, ?)", (k, int(ts)))

    def is_seen(self, *, k: str) -> bool:
        with self._connect() as c:
            row = c.execute("SELECT 1 FROM seen_trades WHERE k = ? LIMIT 1", (k,)).fetchone()
        return bool(row)

    def prune_seen(self, *, min_ts: int) -> int:
        with self._connect() as c:
            cur = c.execute("DELETE FROM seen_trades WHERE ts < ?", (int(min_ts),))
        return int(cur.rowcount or 0)

    def get_market_mapping(self, *, condition_id: str, outcome_index: int) -> MarketMapping | None:
        with self._connect() as c:
            row = c.execute(
                "SELECT * FROM market_map WHERE condition_id = ? AND outcome_index = ?",
                (str(condition_id), int(outcome_index)),
            ).fetchone()
        if not row:
            return None
        return MarketMapping(
            condition_id=str(row["condition_id"]),
            outcome_index=int(row["outcome_index"]),
            token_id=str(row["token_id"]),
            question=str(row["question"]),
            minimum_order_size=float(row["minimum_order_size"]),
            tick_size=float(row["tick_size"]),
            neg_risk=bool(int(row["neg_risk"])),
        )

    def upsert_market_mapping(self, m: MarketMapping) -> None:
        now = int(time.time())
        with self._connect() as c:
            c.execute(
                """
                INSERT INTO market_map(
                  condition_id, outcome_index, token_id, question, minimum_order_size, tick_size, neg_risk, updated_at
                )
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(condition_id, outcome_index) DO UPDATE SET
                  token_id=excluded.token_id,
                  question=excluded.question,
                  minimum_order_size=excluded.minimum_order_size,
                  tick_size=excluded.tick_size,
                  neg_risk=excluded.neg_risk,
                  updated_at=excluded.updated_at
                """,
                (
                    m.condition_id,
                    int(m.outcome_index),
                    m.token_id,
                    m.question,
                    float(m.minimum_order_size),
                    float(m.tick_size),
                    1 if m.neg_risk else 0,
                    now,
                ),
            )

    def get_ledger_state(self) -> LedgerState:
        with self._connect() as c:
            row = c.execute("SELECT cash_usdc, realized_pnl_usdc FROM ledger_state WHERE id = 1").fetchone()
        return LedgerState(cash_usdc=float(row["cash_usdc"]), realized_pnl_usdc=float(row["realized_pnl_usdc"]))

    def set_initial_cash_if_zero(self, cash_usdc: float) -> bool:
        # 仅在初次启动且现金为 0 时初始化，避免二次运行覆盖账本
        with self._connect() as c:
            row = c.execute("SELECT cash_usdc FROM ledger_state WHERE id = 1").fetchone()
            if not row:
                return False
            if float(row["cash_usdc"]) != 0.0:
                return False
            now = int(time.time())
            c.execute(
                "UPDATE ledger_state SET cash_usdc = ?, updated_at = ? WHERE id = 1",
                (float(cash_usdc), now),
            )
        return True

    def list_positions(self) -> list[LedgerPosition]:
        with self._connect() as c:
            rows = c.execute("SELECT * FROM ledger_positions WHERE shares > 0").fetchall()
        return [
            LedgerPosition(
                token_id=str(r["token_id"]),
                condition_id=str(r["condition_id"]),
                outcome_index=int(r["outcome_index"]),
                shares=float(r["shares"]),
                avg_price=float(r["avg_price"]),
            )
            for r in rows
        ]

    def get_position(self, token_id: str) -> LedgerPosition | None:
        with self._connect() as c:
            r = c.execute("SELECT * FROM ledger_positions WHERE token_id = ?", (str(token_id),)).fetchone()
        if not r:
            return None
        return LedgerPosition(
            token_id=str(r["token_id"]),
            condition_id=str(r["condition_id"]),
            outcome_index=int(r["outcome_index"]),
            shares=float(r["shares"]),
            avg_price=float(r["avg_price"]),
        )

    def apply_shadow_fill(
        self,
        *,
        token_id: str,
        condition_id: str,
        outcome_index: int,
        side: str,
        shares: float,
        price: float,
    ) -> dict[str, Any]:
        side_u = str(side).upper()
        if shares <= 0 or price <= 0:
            raise ValueError("shares/price 必须为正数")

        now = int(time.time())
        with self._connect() as c:
            st = c.execute("SELECT cash_usdc, realized_pnl_usdc FROM ledger_state WHERE id = 1").fetchone()
            cash = float(st["cash_usdc"])
            realized = float(st["realized_pnl_usdc"])

            r = c.execute("SELECT shares, avg_price FROM ledger_positions WHERE token_id = ?", (str(token_id),)).fetchone()
            cur_shares = float(r["shares"]) if r else 0.0
            avg_price = float(r["avg_price"]) if r else 0.0

            if side_u == "BUY":
                cost = shares * price
                if cash + 1e-9 < cost:
                    raise RuntimeError(f"现金不足: cash={cash:.6f} cost={cost:.6f}")
                new_cash = cash - cost
                new_shares = cur_shares + shares
                new_avg = ((cur_shares * avg_price) + (shares * price)) / max(1e-12, new_shares)

                c.execute(
                    "UPDATE ledger_state SET cash_usdc=?, realized_pnl_usdc=?, updated_at=? WHERE id=1",
                    (float(new_cash), float(realized), now),
                )
                c.execute(
                    """
                    INSERT INTO ledger_positions(token_id, condition_id, outcome_index, shares, avg_price, updated_at)
                    VALUES (?,?,?,?,?,?)
                    ON CONFLICT(token_id) DO UPDATE SET
                      condition_id=excluded.condition_id,
                      outcome_index=excluded.outcome_index,
                      shares=excluded.shares,
                      avg_price=excluded.avg_price,
                      updated_at=excluded.updated_at
                    """,
                    (str(token_id), str(condition_id), int(outcome_index), float(new_shares), float(new_avg), now),
                )
                return {
                    "side": "BUY",
                    "shares": float(shares),
                    "price": float(price),
                    "cash_delta": -float(cost),
                    "realized_pnl_delta": 0.0,
                    "new_cash": float(new_cash),
                    "new_shares": float(new_shares),
                    "new_avg_price": float(new_avg),
                }

            if side_u == "SELL":
                sell_shares = min(float(shares), float(cur_shares))
                if sell_shares <= 0:
                    raise RuntimeError("无可卖出 shares")
                proceeds = sell_shares * price
                pnl = (price - avg_price) * sell_shares
                new_cash = cash + proceeds
                new_realized = realized + pnl
                new_shares = cur_shares - sell_shares
                new_avg = avg_price if new_shares > 0 else 0.0

                c.execute(
                    "UPDATE ledger_state SET cash_usdc=?, realized_pnl_usdc=?, updated_at=? WHERE id=1",
                    (float(new_cash), float(new_realized), now),
                )
                if new_shares > 0:
                    c.execute(
                        "UPDATE ledger_positions SET shares=?, avg_price=?, updated_at=? WHERE token_id=?",
                        (float(new_shares), float(new_avg), now, str(token_id)),
                    )
                else:
                    c.execute("DELETE FROM ledger_positions WHERE token_id=?", (str(token_id),))
                return {
                    "side": "SELL",
                    "shares": float(sell_shares),
                    "price": float(price),
                    "cash_delta": float(proceeds),
                    "realized_pnl_delta": float(pnl),
                    "new_cash": float(new_cash),
                    "new_shares": float(new_shares),
                    "new_avg_price": float(new_avg),
                }

            raise ValueError(f"unknown side: {side}")

    def add_equity_snapshot(
        self,
        *,
        ts: int,
        cash_usdc: float,
        equity_usdc: float,
        total_cost_exposure_usdc: float,
        unrealized_pnl_usdc: float,
    ) -> None:
        with self._connect() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO equity_snapshots(ts, cash_usdc, equity_usdc, total_cost_exposure_usdc, unrealized_pnl_usdc)
                VALUES (?,?,?,?,?)
                """,
                (int(ts), float(cash_usdc), float(equity_usdc), float(total_cost_exposure_usdc), float(unrealized_pnl_usdc)),
            )

    def dump_debug(self) -> str:
        st = self.get_ledger_state()
        pos = self.list_positions()
        return json.dumps({"ledger": st.__dict__, "positions": [p.__dict__ for p in pos]}, ensure_ascii=False, indent=2)

