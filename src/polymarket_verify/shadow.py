from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .clob_public import ClobPublicClient
from .config_loader import AppConfig
from .data_api import DataApiClient
from .models import ActivityTrade
from .state_db import MarketMapping, StateDB
from .stats import summarize


@dataclass(frozen=True)
class MergedTrade:
    condition_id: str
    outcome_index: int
    asset: str
    side: str
    ts_first: int
    ts_last: int
    n: int
    total_shares: float
    total_usdc: float
    avg_price: float
    tx_hashes: list[str]
    outcome: Optional[str]
    title: Optional[str]


def _trade_key(t: ActivityTrade) -> str:
    # 避免 float 字符串差异导致去重失效，统一小数位
    return "|".join(
        [
            str(t.transactionHash),
            str(t.asset),
            str(t.side),
            f"{float(t.usdcSize):.8f}",
            str(int(t.timestamp)),
        ]
    )


def _now_utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _merge_trades(trades_asc: list[ActivityTrade], *, merge_window_sec: int) -> list[MergedTrade]:
    merged: list[MergedTrade] = []
    cur: list[ActivityTrade] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        condition_id = cur[0].conditionId
        outcome_index = cur[0].outcomeIndex
        asset = cur[0].asset
        side = cur[0].side
        ts_first = int(cur[0].timestamp)
        ts_last = int(cur[-1].timestamp)
        total_shares = float(sum(float(x.size) for x in cur))
        total_usdc = float(sum(float(x.usdcSize) for x in cur))
        w = sum(float(x.size) for x in cur) or 1e-12
        avg_price = float(sum(float(x.price) * float(x.size) for x in cur) / w)
        tx_hashes = [str(x.transactionHash) for x in cur]
        merged.append(
            MergedTrade(
                condition_id=str(condition_id),
                outcome_index=int(outcome_index),
                asset=str(asset),
                side=str(side),
                ts_first=ts_first,
                ts_last=ts_last,
                n=len(cur),
                total_shares=total_shares,
                total_usdc=total_usdc,
                avg_price=avg_price,
                tx_hashes=tx_hashes,
                outcome=cur[0].outcome,
                title=cur[0].title,
            )
        )
        cur = []

    for t in trades_asc:
        if not cur:
            cur = [t]
            continue
        last = cur[-1]
        same_key = (
            t.conditionId == last.conditionId
            and int(t.outcomeIndex) == int(last.outcomeIndex)
            and str(t.side) == str(last.side)
            and str(t.asset) == str(last.asset)
        )
        gap = int(t.timestamp) - int(last.timestamp)
        if same_key and gap <= int(merge_window_sec):
            cur.append(t)
        else:
            flush()
            cur = [t]
    flush()
    return merged


def _compute_exposure(db: StateDB) -> tuple[float, dict[str, float]]:
    total = 0.0
    by_market: dict[str, float] = {}
    for p in db.list_positions():
        cost = float(p.shares) * float(p.avg_price)
        total += cost
        by_market[p.condition_id] = by_market.get(p.condition_id, 0.0) + cost
    return float(total), by_market


def _compute_equity(db: StateDB, clob: ClobPublicClient) -> dict[str, float]:
    st = db.get_ledger_state()
    cash = float(st.cash_usdc)
    total_cost, _ = _compute_exposure(db)
    pos_value = 0.0
    unreal = 0.0
    for p in db.list_positions():
        try:
            _buy, best_sell = clob.get_best_prices(p.token_id)
        except Exception:
            # 标价失败时保守处理：按成本价估值，不把未实现收益算进去
            best_sell = float(p.avg_price)
        v = float(p.shares) * float(best_sell)
        pos_value += v
        unreal += float(p.shares) * (float(best_sell) - float(p.avg_price))
    equity = cash + pos_value
    return {
        "cash_usdc": cash,
        "equity_usdc": float(equity),
        "total_cost_exposure_usdc": float(total_cost),
        "unrealized_pnl_usdc": float(unreal),
        "realized_pnl_usdc": float(st.realized_pnl_usdc),
    }


def _write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _render_kpi_md(*, kpi: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("## Shadow trading KPI")
    lines.append("")
    lines.append(f"- UTC date: `{kpi.get('utc_date')}`")
    lines.append(f"- Decisions: follow={kpi.get('follow',0)} skip={kpi.get('skip',0)} merged_groups={kpi.get('groups',0)}")
    lines.append(f"- Last equity: {kpi.get('last_equity_usdc',0):.4f} USDC (cash={kpi.get('last_cash_usdc',0):.4f})")
    lines.append("")
    lines.append("### Skip reasons")
    lines.append("")
    for k, v in sorted((kpi.get("skip_reasons") or {}).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("### Slippage (abs)")
    lines.append("")
    slip = kpi.get("slippage_abs_summary") or {}
    if slip:
        lines.append(f"- min/p50/p90/p99/max: {slip.get('min')}/{slip.get('p50')}/{slip.get('p90')}/{slip.get('p99')}/{slip.get('max')}")
    else:
        lines.append("- (no data)")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_shadow(*, cfg: AppConfig, out_dir: Path, state_db_path: Path, duration_sec: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    db = StateDB(state_db_path)
    db.set_initial_cash_if_zero(cfg.shadow_initial_cash_usdc)

    api = DataApiClient(base_url=cfg.data_api_base, timeout_sec=cfg.timeout_sec)
    clob = ClobPublicClient(timeout_sec=cfg.timeout_sec)

    start_ts = int(time.time())
    end_ts = start_ts + int(duration_sec)
    log_path = out_dir / f"shadow_{cfg.username}_{start_ts}.jsonl"
    kpi_json_path = out_dir / "shadow_kpi.json"
    kpi_md_path = out_dir / "shadow_kpi.md"

    # KPI in-memory
    follow = 0
    skip = 0
    groups = 0
    skip_reasons: dict[str, int] = {}
    slippage_abs_samples: list[float] = []
    last_kpi_write_at = 0

    # circuit breaker: daily start equity
    daily_key = f"daily_start_equity:{_now_utc_date()}"
    daily_start = db.get_kv(daily_key)
    last_equity = _compute_equity(db, clob)
    if daily_start is None:
        db.set_kv(daily_key, f"{float(last_equity['equity_usdc']):.12f}")

    last_snapshot_at = 0
    snapshot_interval_sec = 300

    # 首次启动默认“从现在开始跟随”，避免把历史成交当作实时信号
    last_seen_key = "shadow_last_seen_ts"
    last_seen_ts = db.get_kv(last_seen_key)
    if last_seen_ts is None:
        # 留一点 overlap，避免启动瞬间漏掉刚发生的一笔
        last_seen_ts = str(int(time.time()) - 60)
        db.set_kv(last_seen_key, last_seen_ts)
    last_seen = int(last_seen_ts)

    _write_jsonl(
        log_path,
        {
            "type": "start",
            "ts": start_ts,
            "end_ts": end_ts,
            "cfg": {
                "initial_cash_usdc": cfg.shadow_initial_cash_usdc,
                "k": cfg.shadow_k,
                "min_per_trade_usdc": cfg.shadow_min_per_trade_usdc,
                "max_per_trade_usdc": cfg.shadow_max_per_trade_usdc,
                "max_abs_slippage": cfg.shadow_max_abs_slippage,
                "max_market_exposure_usdc": cfg.shadow_max_market_exposure_usdc,
                "max_total_exposure_usdc": cfg.shadow_max_total_exposure_usdc,
                "daily_loss_limit_usdc": cfg.shadow_daily_loss_limit_usdc,
                "merge_window_sec": cfg.shadow_merge_window_sec,
                "poll_interval_sec": cfg.shadow_poll_interval_sec,
                "jitter_sec": cfg.shadow_jitter_sec,
                "fetch_limit": cfg.shadow_fetch_limit,
                "condition_whitelist": cfg.shadow_condition_whitelist,
            },
            "state_db": str(state_db_path),
            "shadow_last_seen_ts": last_seen,
        },
    )

    while True:
        now = int(time.time())
        if now >= end_ts:
            break

        # daily boundary (UTC)
        daily_key_now = f"daily_start_equity:{_now_utc_date()}"
        if daily_key_now != daily_key:
            daily_key = daily_key_now
            if db.get_kv(daily_key) is None:
                last_equity = _compute_equity(db, clob)
                db.set_kv(daily_key, f"{float(last_equity['equity_usdc']):.12f}")

        if now - last_snapshot_at >= snapshot_interval_sec:
            last_equity = _compute_equity(db, clob)
            db.add_equity_snapshot(
                ts=now,
                cash_usdc=float(last_equity["cash_usdc"]),
                equity_usdc=float(last_equity["equity_usdc"]),
                total_cost_exposure_usdc=float(last_equity["total_cost_exposure_usdc"]),
                unrealized_pnl_usdc=float(last_equity["unrealized_pnl_usdc"]),
            )
            _write_jsonl(log_path, {"type": "snapshot", "ts": now, "equity": last_equity})
            last_snapshot_at = now

        # circuit breaker
        daily_start_v = float(db.get_kv(daily_key, "0") or 0.0)
        daily_loss = max(0.0, daily_start_v - float(last_equity["equity_usdc"]))
        circuit = daily_loss > float(cfg.shadow_daily_loss_limit_usdc)

        # fetch latest trades and filter new
        try:
            raw = api.get_activity(user=cfg.address, limit=cfg.shadow_fetch_limit, offset=0, types="TRADE")
            trades = [ActivityTrade.model_validate(x) for x in raw if x.get("type") == "TRADE"]
        except Exception as e:
            _write_jsonl(log_path, {"type": "error", "ts": now, "where": "data_api.get_activity", "err": repr(e)})
            time.sleep(max(0.5, cfg.shadow_poll_interval_sec))
            continue

        # sort asc for deterministic processing
        trades.sort(key=lambda x: (int(x.timestamp), str(x.transactionHash)))

        fresh: list[ActivityTrade] = []
        ttl_min_ts = int(now - cfg.shadow_dedupe_ttl_sec)
        pruned = db.prune_seen(min_ts=ttl_min_ts)
        if pruned:
            _write_jsonl(log_path, {"type": "housekeep", "ts": now, "pruned_seen": pruned, "min_ts": ttl_min_ts})

        for t in trades:
            if int(t.timestamp) <= last_seen:
                continue
            k = _trade_key(t)
            if db.is_seen(k=k):
                continue
            db.add_seen(k=k, ts=int(t.timestamp))
            fresh.append(t)

        if not fresh:
            time.sleep(max(0.5, cfg.shadow_poll_interval_sec + random.uniform(-cfg.shadow_jitter_sec, cfg.shadow_jitter_sec)))
            continue

        last_seen = max(last_seen, max(int(t.timestamp) for t in fresh))
        db.set_kv(last_seen_key, str(int(last_seen)))

        merged = _merge_trades(fresh, merge_window_sec=cfg.shadow_merge_window_sec)
        groups += len(merged)

        for mt in merged:
            if int(time.time()) >= end_ts:
                break
            decision_ts = int(time.time())
            reason: str | None = None

            if cfg.shadow_condition_whitelist and mt.condition_id not in set(cfg.shadow_condition_whitelist):
                reason = "NOT_IN_WHITELIST"

            if circuit and reason is None:
                reason = "DAILY_LOSS_CIRCUIT_BREAKER"

            mapping = None
            if reason is None:
                mapping = db.get_market_mapping(condition_id=mt.condition_id, outcome_index=mt.outcome_index)
                if mapping is None:
                    try:
                        mi = clob.get_market_info(mt.condition_id)
                        token_id = clob.resolve_token_id(
                            condition_id=mt.condition_id,
                            outcome_index=mt.outcome_index,
                            outcome=mt.outcome,
                        )
                        mapping = MarketMapping(
                            condition_id=mt.condition_id,
                            outcome_index=int(mt.outcome_index),
                            token_id=str(token_id),
                            question=str(mi.question),
                            minimum_order_size=float(mi.minimum_order_size),
                            tick_size=float(mi.tick_size),
                            neg_risk=bool(mi.neg_risk),
                        )
                        db.upsert_market_mapping(mapping)
                    except Exception as e:
                        reason = f"MAP_TOKEN_FAILED:{type(e).__name__}"

            best_buy = best_sell = None
            if reason is None and mapping is not None:
                try:
                    best_buy, best_sell = clob.get_best_prices(mapping.token_id)
                except Exception as e:
                    reason = f"QUOTE_FAILED:{type(e).__name__}"

            # sizing
            his_cost = abs(float(mt.total_usdc)) if float(mt.total_usdc) else abs(float(mt.total_shares) * float(mt.avg_price))
            you_cost = min(float(cfg.shadow_k) * float(his_cost), float(cfg.shadow_max_per_trade_usdc))
            if reason is None and you_cost < float(cfg.shadow_min_per_trade_usdc):
                reason = "TOO_SMALL"

            # exposure limits
            total_cost_exposure, by_market = _compute_exposure(db)
            market_cost_exposure = float(by_market.get(mt.condition_id, 0.0))
            if reason is None and (total_cost_exposure + you_cost) > float(cfg.shadow_max_total_exposure_usdc) and mt.side.upper() == "BUY":
                reason = "TOTAL_EXPOSURE_LIMIT"
            if reason is None and (market_cost_exposure + you_cost) > float(cfg.shadow_max_market_exposure_usdc) and mt.side.upper() == "BUY":
                reason = "MARKET_EXPOSURE_LIMIT"

            # slippage & executable price
            exec_price = None
            slip_abs = None
            if reason is None and best_buy is not None and best_sell is not None:
                exec_price = float(best_buy) if mt.side.upper() == "BUY" else float(best_sell)
                slip_abs = abs(float(exec_price) - float(mt.avg_price))
                slippage_abs_samples.append(float(slip_abs))
                if float(slip_abs) > float(cfg.shadow_max_abs_slippage):
                    reason = "SLIPPAGE_TOO_HIGH"

            # simulate fill
            fill = None
            if reason is None and mapping is not None and exec_price is not None:
                if mt.side.upper() == "BUY":
                    shares = you_cost / max(1e-12, float(exec_price))
                    st = db.get_ledger_state()
                    if float(st.cash_usdc) + 1e-9 < you_cost:
                        reason = "INSUFFICIENT_CASH"
                    else:
                        fill = db.apply_shadow_fill(
                            token_id=mapping.token_id,
                            condition_id=mt.condition_id,
                            outcome_index=int(mt.outcome_index),
                            side="BUY",
                            shares=float(shares),
                            price=float(exec_price),
                        )
                else:
                    pos = db.get_position(mapping.token_id)
                    if not pos or float(pos.shares) <= 0:
                        reason = "NO_POSITION_TO_SELL"
                    else:
                        shares = you_cost / max(1e-12, float(exec_price))
                        fill = db.apply_shadow_fill(
                            token_id=mapping.token_id,
                            condition_id=mt.condition_id,
                            outcome_index=int(mt.outcome_index),
                            side="SELL",
                            shares=float(shares),
                            price=float(exec_price),
                        )

            if reason is None:
                follow += 1
                decision = "FOLLOW"
            else:
                skip += 1
                decision = "SKIP"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

            _write_jsonl(
                log_path,
                {
                    "type": "decision",
                    "ts": decision_ts,
                    "decision": decision,
                    "reason": reason,
                    "his": {
                        "condition_id": mt.condition_id,
                        "outcome_index": mt.outcome_index,
                        "side": mt.side,
                        "avg_price": mt.avg_price,
                        "total_usdc": mt.total_usdc,
                        "total_shares": mt.total_shares,
                        "n": mt.n,
                        "ts_first": mt.ts_first,
                        "ts_last": mt.ts_last,
                        "tx_hashes": mt.tx_hashes,
                        "title": mt.title,
                    },
                    "map": mapping.__dict__ if mapping else None,
                    "quote": {"best_buy": best_buy, "best_sell": best_sell, "exec_price": exec_price, "slip_abs": slip_abs},
                    "you": {"his_cost_usdc": his_cost, "you_cost_usdc": you_cost, "k": cfg.shadow_k},
                    "fill": fill,
                    "ledger": db.get_ledger_state().__dict__,
                },
            )

            # KPI 增量写：避免“处理一大批历史成交”时长时间没输出
            if int(time.time()) - last_kpi_write_at >= 10:
                slip_summary = summarize(slippage_abs_samples) if slippage_abs_samples else {}
                last_equity = _compute_equity(db, clob)
                kpi = {
                    "utc_date": _now_utc_date(),
                    "follow": follow,
                    "skip": skip,
                    "groups": groups,
                    "skip_reasons": skip_reasons,
                    "slippage_abs_summary": slip_summary,
                    "last_cash_usdc": float(last_equity["cash_usdc"]),
                    "last_equity_usdc": float(last_equity["equity_usdc"]),
                    "shadow_last_seen_ts": int(last_seen),
                }
                kpi_json_path.write_text(json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")
                kpi_md_path.write_text(_render_kpi_md(kpi=kpi), encoding="utf-8")
                last_kpi_write_at = int(time.time())

        time.sleep(max(0.5, cfg.shadow_poll_interval_sec + random.uniform(-cfg.shadow_jitter_sec, cfg.shadow_jitter_sec)))

    # 结束时写一次最终 KPI（即使中途没触发增量写）
    slip_summary = summarize(slippage_abs_samples) if slippage_abs_samples else {}
    last_equity = _compute_equity(db, clob)
    kpi = {
        "utc_date": _now_utc_date(),
        "follow": follow,
        "skip": skip,
        "groups": groups,
        "skip_reasons": skip_reasons,
        "slippage_abs_summary": slip_summary,
        "last_cash_usdc": float(last_equity["cash_usdc"]),
        "last_equity_usdc": float(last_equity["equity_usdc"]),
        "shadow_last_seen_ts": int(last_seen),
    }
    kpi_json_path.write_text(json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")
    kpi_md_path.write_text(_render_kpi_md(kpi=kpi), encoding="utf-8")

    _write_jsonl(log_path, {"type": "end", "ts": int(time.time()), "kpi": kpi})
    return 0

