from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from dotenv import load_dotenv

from .clob_public import ClobPublicClient
from .config_loader import AppConfig
from .data_api import DataApiClient
from .executor import Action, ExecResult, LiveExecutor, ShadowExecutor
from .models import ActivityTrade
from .state_db import MarketMapping, StateDB
from .stats import max_drawdown, summarize


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


def _get_live_collateral_balance_usdc() -> tuple[float | None, str | None]:
    """读取 CLOB collateral 余额（USDC），失败返回错误原因。"""
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds, AssetType, BalanceAllowanceParams
    except Exception as e:
        return None, f"CLOB_IMPORT_ERROR:{type(e).__name__}"

    pk = os.getenv("PRIVATE_KEY") or ""
    funder = os.getenv("FUNDER_ADDRESS") or ""
    sig_type = int(os.getenv("SIGNATURE_TYPE", "0"))
    if not pk:
        return None, "MISSING_PRIVATE_KEY"
    if not funder:
        return None, "MISSING_FUNDER_ADDRESS"

    api_key = os.getenv("POLY_API_KEY") or ""
    api_secret = os.getenv("POLY_SECRET") or ""
    api_pass = os.getenv("POLY_PASSPHRASE") or ""
    creds = None
    if api_key and api_secret and api_pass:
        creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_pass)

    try:
        client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=pk,
            creds=creds,
            signature_type=sig_type,
            funder=funder,
        )
        if creds is None:
            derived = client.create_or_derive_api_creds()
            client.set_api_creds(derived)

        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, token_id=None, signature_type=sig_type)
        bal = client.get_balance_allowance(params)
        raw = bal.get("balance") if isinstance(bal, dict) else getattr(bal, "balance", None)
        if raw is None:
            return None, "CLOB_BALANCE_MISSING"
        return float(raw) / 1_000_000.0, None
    except Exception as e:
        return None, f"CLOB_BALANCE_ERROR:{type(e).__name__}"


def _quantize_down(price: float, tick: float) -> float:
    if tick <= 0:
        return float(price)
    n = int(float(price) / float(tick) + 1e-12)
    return float(n * float(tick))


def _quantize_up(price: float, tick: float) -> float:
    if tick <= 0:
        return float(price)
    n = int(float(price) / float(tick) + 1 - 1e-12)
    return float(n * float(tick))


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


def _apply_confirmed_live_fill(*, exec_shadow: ShadowExecutor, base_action: Action, exec_result: ExecResult) -> bool:
    # 仅将“已确认成交”的份额记入账本，避免把“挂单成功但未成交”误记为持仓/现金变化
    filled_shares = float(exec_result.filled_shares)
    fill_price = float(exec_result.avg_fill_price)
    if filled_shares <= 1e-12 or fill_price <= 0:
        return False
    fill_usdc = float(exec_result.filled_usdc) if float(exec_result.filled_usdc) > 0 else float(filled_shares * fill_price)
    filled_action = Action(
        run_id=int(base_action.run_id),
        seen_ts=int(base_action.seen_ts),
        trade_ts=int(base_action.trade_ts),
        condition_id=str(base_action.condition_id),
        outcome_index=int(base_action.outcome_index),
        token_id=str(base_action.token_id),
        side=str(base_action.side),
        usdc=float(fill_usdc),
        shares=float(filled_shares),
        price=float(fill_price),
        tick_size=float(base_action.tick_size),
        neg_risk=bool(base_action.neg_risk),
    )
    exec_shadow.execute(filled_action)
    return True


def _refresh_market_mapping(
    *,
    clob: ClobPublicClient,
    db: StateDB,
    condition_id: str,
    outcome_index: int,
    outcome: str | None,
) -> MarketMapping:
    mi = clob.get_market_info(condition_id)
    token_id = clob.resolve_token_id(
        condition_id=condition_id,
        outcome_index=outcome_index,
        outcome=outcome,
    )
    mapping = MarketMapping(
        condition_id=str(condition_id),
        outcome_index=int(outcome_index),
        token_id=str(token_id),
        question=str(mi.question),
        minimum_order_size=float(mi.minimum_order_size),
        tick_size=float(mi.tick_size),
        neg_risk=bool(mi.neg_risk),
    )
    db.upsert_market_mapping(mapping)
    return mapping


def _render_kpi_md(*, kpi: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("## 跟随交易实时报告（本次运行）")
    lines.append("")
    lines.append(f"- 运行ID: `{kpi.get('run_id')}`")
    lines.append(f"- 模式: **{kpi.get('mode','SHADOW')}**（只有 LIVE 才会真实发单）")
    lines.append(f"- UTC 日期: `{kpi.get('utc_date')}`")
    lines.append(f"- 决策统计: follow={kpi.get('follow',0)} skip={kpi.get('skip',0)} 合并组数={kpi.get('groups',0)}")
    lines.append("")
    lines.append("### 资金与回撤")
    lines.append("")
    lines.append(
        f"- 当前权益: **{kpi.get('equity_now_usdc',0):.4f}** USDC（现金 {kpi.get('cash_now_usdc',0):.4f}）"
    )
    lines.append(
        f"- 本次收益: **{kpi.get('pnl_usdc',0):.4f}** USDC（收益率 {kpi.get('pnl_pct',0):.2f}%）"
    )
    lines.append(
        f"- 最大回撤(MDD): **{kpi.get('mdd_abs_usdc',0):.4f}** USDC（{kpi.get('mdd_pct',0):.2f}%）"
    )
    lines.append("")
    lines.append("### 未成交/未执行原因（Top）")
    lines.append("")
    reasons = kpi.get("skip_reasons") or {}
    if reasons:
        for k, v in sorted(reasons.items(), key=lambda x: (-x[1], x[0]))[:12]:
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- （暂无）")
    lines.append("")
    lines.append("### 实盘延迟（seen→ack）")
    lines.append("")
    lat = kpi.get("live_seen_to_ack_ms") or {}
    if lat and lat.get("count"):
        lines.append(
            f"- 样本数: {lat.get('count')}，p50/p90/p99/max: {lat.get('p50')}/{lat.get('p90')}/{lat.get('p99')}/{lat.get('max')} ms"
        )
    else:
        lines.append("- （暂无 live 样本）")
    lines.append("")
    lines.append("### 滑点（abs，影子口径）")
    lines.append("")
    slip = kpi.get("slippage_abs_summary") or {}
    if slip and slip.get("count"):
        lines.append(
            f"- 样本数: {slip.get('count')}，min/p50/p90/p99/max: {slip.get('min')}/{slip.get('p50')}/{slip.get('p90')}/{slip.get('p99')}/{slip.get('max')}"
        )
    else:
        lines.append("- （暂无）")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_shadow(
    *,
    cfg: AppConfig,
    out_dir: Path,
    state_db_path: Path,
    duration_sec: int,
    live: bool = False,
    taker: bool = False,
    dual: bool = False,
    live_state_db_path: Path | None = None,
    reset_state: bool = False,
    sync_live_cash: bool = False,
) -> int:
    if dual and not live:
        live = True  # dual 必须同时跑 live
    out_dir.mkdir(parents=True, exist_ok=True)

    path_live: Path | None = None
    if dual:
        path_live = live_state_db_path or (state_db_path.parent / f"{state_db_path.stem}_live{state_db_path.suffix}")

    if reset_state:
        targets = [state_db_path]
        if path_live is not None:
            targets.append(path_live)
        for p in targets:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                # 删除失败时保持原状态继续运行，避免启动直接中断
                pass

    initial_cash_shadow = float(cfg.shadow_initial_cash_usdc)
    live_initial_cash = initial_cash_shadow
    live_initial_cash_source: str = "config"
    live_initial_cash_warn: str | None = None
    api = DataApiClient(base_url=cfg.data_api_base, timeout_sec=cfg.timeout_sec)
    clob = ClobPublicClient(timeout_sec=cfg.timeout_sec)
    if live or dual:
        load_dotenv(".env")
        funder = (os.getenv("FUNDER_ADDRESS") or "").strip()
        clob_cash, clob_err = _get_live_collateral_balance_usdc()
        if clob_cash is not None and float(clob_cash) > 0:
            live_initial_cash = float(clob_cash)
            live_initial_cash_source = "clob_collateral"
        elif funder:
            try:
                val = api.get_value(user=str(funder).lower())
                if val is not None and float(val) > 0:
                    live_initial_cash = float(val)
                    live_initial_cash_source = "data_api"
                else:
                    live_initial_cash_source = "config_data_api_empty"
                    live_initial_cash_warn = (
                        "CLOB/Data API 余额均不可用，live 初始资金回退到 config.shadow.initial_cash_usdc"
                        + (f" (clob={clob_err})" if clob_err else "")
                    )
            except Exception as e:
                live_initial_cash_source = "config_data_api_error"
                live_initial_cash_warn = (
                    f"CLOB 余额不可用且 Data API 读取失败，live 初始资金回退到配置值: {type(e).__name__}"
                    + (f" (clob={clob_err})" if clob_err else "")
                )
        else:
            live_initial_cash_source = "config_missing_funder"
            live_initial_cash_warn = (
                "未设置 FUNDER_ADDRESS，live 初始资金回退到 config.shadow.initial_cash_usdc"
                + (f" (clob={clob_err})" if clob_err else "")
            )

    db = StateDB(state_db_path)
    if dual:
        if sync_live_cash:
            # dual 对比要同一起跑线：shadow 与 live 都同步为同一实时现金
            db.force_set_cash(live_initial_cash)
        else:
            db.set_initial_cash_if_zero(initial_cash_shadow)
    elif live:
        if sync_live_cash:
            db.force_set_cash(live_initial_cash)
        else:
            db.set_initial_cash_if_zero(live_initial_cash)
    else:
        db.set_initial_cash_if_zero(initial_cash_shadow)

    db_live: StateDB | None = None
    if dual:
        assert path_live is not None
        db_live = StateDB(path_live)
        if sync_live_cash:
            db_live.force_set_cash(live_initial_cash)
        else:
            db_live.set_initial_cash_if_zero(live_initial_cash)

    whitelist = set(cfg.shadow_condition_whitelist)
    run_id = int(time.time())

    shadow_exec = ShadowExecutor(db=db)
    live_exec: LiveExecutor | None = None
    if live:
        live_exec = LiveExecutor(
            data_api_base=cfg.data_api_base,
            timeout_sec=cfg.timeout_sec,
            post_only=bool(cfg.live_post_only),
            taker=bool(taker or cfg.live_taker),
            order_type=str(cfg.live_order_type),
            cancel_after_sec=int(cfg.live_cancel_after_sec),
            slippage=float(cfg.live_slippage),
        )
    shadow_exec_live: ShadowExecutor | None = None
    if dual and db_live is not None:
        shadow_exec_live = ShadowExecutor(db=db_live)

    start_ts = int(run_id)
    end_ts = start_ts + int(duration_sec)
    report_prefix = "dual" if dual else ("live" if live else "shadow")
    log_path = out_dir / f"{report_prefix}_{cfg.username}_{start_ts}.jsonl"
    kpi_json_path = out_dir / f"{report_prefix}_kpi.json"
    kpi_md_path = out_dir / f"{report_prefix}_kpi.md"
    if dual:
        kpi_shadow_json = out_dir / "shadow_kpi.json"
        kpi_shadow_md = out_dir / "shadow_kpi.md"
        kpi_live_json = out_dir / "live_kpi.json"
        kpi_live_md = out_dir / "live_kpi.md"

    # KPI in-memory
    follow = 0
    skip = 0
    groups = 0
    skip_reasons: dict[str, int] = {}
    slippage_abs_samples: list[float] = []
    last_kpi_write_at = 0
    # live 风控计数：按“尝试发单”计，保守限制频率与预算
    live_order_attempt_ts: list[int] = []
    live_buy_budget_by_utc_day: dict[str, float] = {}

    # circuit breaker: daily start equity
    daily_key = f"daily_start_equity:{_now_utc_date()}"
    daily_start = db.get_kv(daily_key)
    last_equity = _compute_equity(db, clob)
    if daily_start is None:
        db.set_kv(daily_key, f"{float(last_equity['equity_usdc']):.12f}")
    if dual and db_live is not None:
        eq_live = _compute_equity(db_live, clob)
        if db_live.get_kv(daily_key) is None:
            db_live.set_kv(daily_key, f"{float(eq_live['equity_usdc']):.12f}")

    last_snapshot_at = 0
    snapshot_interval_sec = 300

    # 默认“从现在开始跟随”，避免把历史成交当作实时信号。
    # 注意：如果 state_db 里残留了很旧的 last_seen_ts，也要向前钳制到 now-60，
    # 否则会触发“回放历史成交”，导致你观测到的延迟被历史堆积污染。
    last_seen_key = "shadow_last_seen_ts"
    last_seen_ts = db.get_kv(last_seen_key)
    clamp_min = int(time.time()) - 60
    if last_seen_ts is None:
        # 留一点 overlap，避免启动瞬间漏掉刚发生的一笔
        last_seen_ts = str(clamp_min)
        db.set_kv(last_seen_key, last_seen_ts)
    last_seen = max(int(last_seen_ts), int(clamp_min))
    db.set_kv(last_seen_key, str(int(last_seen)))

    _write_jsonl(
        log_path,
        {
            "type": "start",
            "ts": start_ts,
            "end_ts": end_ts,
            "cfg": {
                "initial_cash_usdc": initial_cash_shadow,
                "live_initial_cash_usdc": live_initial_cash if (live or dual) else None,
                "live_initial_cash_source": live_initial_cash_source if (live or dual) else None,
                "follow_all": bool(cfg.shadow_follow_all),
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
                "live": bool(live),
                "taker": bool(taker),
                "dual": bool(dual),
                "reset_state": bool(reset_state),
                "sync_live_cash": bool(sync_live_cash),
            },
            "state_db": str(state_db_path),
            "live_state_db": str(db_live.path) if db_live is not None else None,
            "shadow_last_seen_ts": last_seen,
        },
    )
    if live_initial_cash_warn and (live or dual):
        _write_jsonl(
            log_path,
            {
                "type": "warn",
                "ts": int(time.time()),
                "where": "live_initial_cash",
                "msg": live_initial_cash_warn,
                "source": live_initial_cash_source,
                "live_initial_cash_usdc": float(live_initial_cash),
            },
        )

    run_start_key = f"run_start_equity:{int(start_ts)}"
    if db.get_kv(run_start_key) is None:
        db.set_kv(run_start_key, f"{float(last_equity['equity_usdc']):.12f}")
    if dual and db_live is not None:
        if db_live.get_kv(run_start_key) is None:
            eq_live = _compute_equity(db_live, clob)
            db_live.set_kv(run_start_key, f"{float(eq_live['equity_usdc']):.12f}")

    def _build_kpi(
        db_equity: StateDB,
        mode_str: str,
        db_live_timings: StateDB | None = None,
    ) -> dict[str, Any]:
        eq = _compute_equity(db_equity, clob)
        equity_now = float(eq["equity_usdc"])
        cash_now = float(eq["cash_usdc"])

        equity_start = float(db_equity.get_kv(run_start_key, f"{equity_now:.12f}") or equity_now)
        pnl = equity_now - equity_start
        pnl_pct = (pnl / max(1e-9, equity_start)) * 100.0

        series = db_equity.list_equity_since(start_ts=start_ts)
        if not series or abs(series[-1] - equity_now) > 1e-9:
            series = list(series) + [equity_now]
        dd = max_drawdown(series)
        peak = max(series) if series else equity_now
        mdd_pct = (dd.mdd_abs / max(1e-9, float(peak))) * 100.0

        db_lt = db_live_timings if db_live_timings is not None else db_equity
        live_ms = db_lt.list_live_seen_to_ack_ms(run_id=int(start_ts), limit=2000)
        live_sum = summarize([float(x) for x in live_ms]) if live_ms else {}

        slip_sum = summarize(slippage_abs_samples) if slippage_abs_samples else {}

        return {
            "run_id": int(start_ts),
            "utc_date": _now_utc_date(),
            "mode": mode_str,
            "follow": int(follow),
            "skip": int(skip),
            "groups": int(groups),
            "skip_reasons": dict(skip_reasons),
            "equity_start_usdc": float(equity_start),
            "equity_now_usdc": float(equity_now),
            "cash_now_usdc": float(cash_now),
            "pnl_usdc": float(pnl),
            "pnl_pct": float(pnl_pct),
            "mdd_abs_usdc": float(dd.mdd_abs),
            "mdd_pct": float(mdd_pct),
            "live_seen_to_ack_ms": live_sum,
            "slippage_abs_summary": slip_sum,
            "shadow_last_seen_ts": int(last_seen),
        }

    # 启动时立即写一份初始 KPI，便于确认运行状态与初始资金（无需等第一笔跟随）
    if dual and db_live is not None:
        kpi_s = _build_kpi(db, "SHADOW", None)
        kpi_l = _build_kpi(db_live, "LIVE", db)
        kpi_shadow_json.write_text(json.dumps(kpi_s, ensure_ascii=False, indent=2), encoding="utf-8")
        kpi_shadow_md.write_text(_render_kpi_md(kpi=kpi_s), encoding="utf-8")
        kpi_live_json.write_text(json.dumps(kpi_l, ensure_ascii=False, indent=2), encoding="utf-8")
        kpi_live_md.write_text(_render_kpi_md(kpi=kpi_l), encoding="utf-8")
    else:
        mode_str = "LIVE" if live else "SHADOW"
        kpi = _build_kpi(db, mode_str, None)
        kpi_json_path.write_text(json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")
        kpi_md_path.write_text(_render_kpi_md(kpi=kpi), encoding="utf-8")

    try:
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
                if dual and db_live is not None and db_live.get_kv(daily_key) is None:
                    eq_live = _compute_equity(db_live, clob)
                    db_live.set_kv(daily_key, f"{float(eq_live['equity_usdc']):.12f}")

            if now - last_snapshot_at >= snapshot_interval_sec:
                last_equity = _compute_equity(db, clob)
                db.add_equity_snapshot(
                    ts=now,
                    cash_usdc=float(last_equity["cash_usdc"]),
                    equity_usdc=float(last_equity["equity_usdc"]),
                    total_cost_exposure_usdc=float(last_equity["total_cost_exposure_usdc"]),
                    unrealized_pnl_usdc=float(last_equity["unrealized_pnl_usdc"]),
                )
                snap_obj: dict[str, Any] = {"type": "snapshot", "ts": now}
                if dual and db_live is not None:
                    eq_live = _compute_equity(db_live, clob)
                    db_live.add_equity_snapshot(
                        ts=now,
                        cash_usdc=float(eq_live["cash_usdc"]),
                        equity_usdc=float(eq_live["equity_usdc"]),
                        total_cost_exposure_usdc=float(eq_live["total_cost_exposure_usdc"]),
                        unrealized_pnl_usdc=float(eq_live["unrealized_pnl_usdc"]),
                    )
                    snap_obj["equity_shadow"] = last_equity
                    snap_obj["equity_live"] = eq_live
                else:
                    snap_obj["equity"] = last_equity
                _write_jsonl(log_path, snap_obj)
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

            # 本轮轮询“首次看到”的时刻：用于 staleness 与（live）seen→ack
            seen_ts = int(time.time())

            for mt in merged:
                if int(time.time()) >= end_ts:
                    break

                decision_ts = int(time.time())
                reason: str | None = None

                if not cfg.shadow_follow_all:
                    if whitelist and mt.condition_id not in whitelist:
                        reason = "NOT_IN_WHITELIST"
                    if circuit and reason is None:
                        reason = "DAILY_LOSS_CIRCUIT_BREAKER"

                mapping = None
                if reason is None:
                    mapping = db.get_market_mapping(condition_id=mt.condition_id, outcome_index=mt.outcome_index)
                    if mapping is None:
                        try:
                            mapping = _refresh_market_mapping(
                                clob=clob,
                                db=db,
                                condition_id=mt.condition_id,
                                outcome_index=mt.outcome_index,
                                outcome=mt.outcome,
                            )
                        except Exception as e:
                            reason = f"MAP_TOKEN_FAILED:{type(e).__name__}"
                    elif live:
                        # live/dual 下优先使用最新 market 元信息，避免旧缓存 tick 触发 invalid tick size
                        try:
                            mapping = _refresh_market_mapping(
                                clob=clob,
                                db=db,
                                condition_id=mt.condition_id,
                                outcome_index=mt.outcome_index,
                                outcome=mt.outcome,
                            )
                        except Exception:
                            # 刷新失败则回退到已有缓存，避免因为临时网络波动整体停摆
                            pass

                best_ask = best_bid = None
                if reason is None and mapping is not None:
                    try:
                        best_ask, best_bid = clob.get_best_prices(mapping.token_id)
                    except Exception:
                        if cfg.shadow_follow_all:
                            best_ask = None
                            best_bid = None
                        else:
                            reason = "QUOTE_FAILED"

                # sizing
                his_cost = abs(float(mt.total_usdc)) if float(mt.total_usdc) else abs(float(mt.total_shares) * float(mt.avg_price))
                you_cost = min(float(cfg.shadow_k) * float(his_cost), float(cfg.shadow_max_per_trade_usdc))
                if not cfg.shadow_follow_all and reason is None and you_cost < float(cfg.shadow_min_per_trade_usdc):
                    reason = "TOO_SMALL"

                # exposure limits (基于影子账本；live 模式初期也沿用该近似)
                total_cost_exposure, by_market = _compute_exposure(db)
                market_cost_exposure = float(by_market.get(mt.condition_id, 0.0))
                if not cfg.shadow_follow_all:
                    if reason is None and (total_cost_exposure + you_cost) > float(cfg.shadow_max_total_exposure_usdc) and mt.side.upper() == "BUY":
                        reason = "TOTAL_EXPOSURE_LIMIT"
                    if reason is None and (market_cost_exposure + you_cost) > float(cfg.shadow_max_market_exposure_usdc) and mt.side.upper() == "BUY":
                        reason = "MARKET_EXPOSURE_LIMIT"

                # exec price & slippage（影子/计划口径）
                exec_price = None
                slip_abs = None
                if reason is None:
                    if best_ask is not None and best_bid is not None:
                        exec_price = float(best_ask) if mt.side.upper() == "BUY" else float(best_bid)
                        slip_abs = abs(float(exec_price) - float(mt.avg_price))
                        slippage_abs_samples.append(float(slip_abs))
                        if (not cfg.shadow_follow_all) and float(slip_abs) > float(cfg.shadow_max_abs_slippage):
                            reason = "SLIPPAGE_TOO_HIGH"
                    elif cfg.shadow_follow_all:
                        exec_price = float(mt.avg_price)
                        slip_abs = 0.0
                        slippage_abs_samples.append(0.0)

                exec_result = None
                action = None

                if reason is None and mapping is not None and exec_price is not None:
                    side = mt.side.upper()
                    shares = you_cost / max(1e-12, float(exec_price))
                    if live:
                        now_ts = int(time.time())
                        hour_start = now_ts - 3600
                        live_order_attempt_ts = [x for x in live_order_attempt_ts if int(x) >= hour_start]
                        if len(live_order_attempt_ts) >= int(cfg.live_max_orders_per_hour):
                            reason = "LIVE_MAX_ORDERS_PER_HOUR"
                        elif side == "BUY":
                            day_key = _now_utc_date()
                            used = float(live_buy_budget_by_utc_day.get(day_key, 0.0))
                            if (used + float(you_cost)) > float(cfg.live_max_usdc_per_day):
                                reason = "LIVE_MAX_USDC_PER_DAY"
                    # live 下单用更安全的 maker 价格；taker 时使用可成交参考价
                    if reason is None and dual and live and live_exec is not None and shadow_exec_live is not None:
                        # dual 模式：先 shadow（模拟无延迟），再 live，成功则同步到 live 账本
                        action_s = Action(
                            run_id=int(start_ts),
                            seen_ts=int(seen_ts),
                            trade_ts=int(mt.ts_last),
                            condition_id=mt.condition_id,
                            outcome_index=int(mt.outcome_index),
                            token_id=str(mapping.token_id),
                            side=side,
                            usdc=float(you_cost),
                            shares=float(shares),
                            price=float(exec_price),
                            tick_size=float(mapping.tick_size),
                            neg_risk=bool(mapping.neg_risk),
                        )
                        shadow_exec.execute(action_s)
                        if side == "BUY":
                            live_price = _quantize_down(float(exec_price) - float(mapping.tick_size), float(mapping.tick_size)) if not (taker or cfg.live_taker) else float(exec_price)
                            live_price = max(float(mapping.tick_size), live_price)
                        else:
                            live_price = _quantize_up(float(exec_price) + float(mapping.tick_size), float(mapping.tick_size)) if not (taker or cfg.live_taker) else float(exec_price)
                        action_l = Action(
                            run_id=int(start_ts),
                            seen_ts=int(seen_ts),
                            trade_ts=int(mt.ts_last),
                            condition_id=mt.condition_id,
                            outcome_index=int(mt.outcome_index),
                            token_id=str(mapping.token_id),
                            side=side,
                            usdc=float(you_cost),
                            shares=float(shares),
                            price=float(live_price),
                            tick_size=float(mapping.tick_size),
                            neg_risk=bool(mapping.neg_risk),
                        )
                        live_order_attempt_ts.append(int(time.time()))
                        if side == "BUY":
                            day_key = _now_utc_date()
                            live_buy_budget_by_utc_day[day_key] = float(live_buy_budget_by_utc_day.get(day_key, 0.0)) + float(you_cost)
                        exec_result = live_exec.execute(action_l)
                        db.add_live_order_timing(
                            run_id=int(start_ts),
                            ts=int(decision_ts),
                            seen_ts=int(seen_ts),
                            trade_ts=int(mt.ts_last),
                            staleness_sec=float(int(seen_ts) - int(mt.ts_last)),
                            side=str(side),
                            condition_id=str(mapping.condition_id),
                            token_id=str(mapping.token_id),
                            usdc=float(you_cost),
                            shares=float(shares),
                            price=float(live_price),
                            ok=bool(exec_result.ok),
                            order_id=exec_result.order_id,
                            reason=exec_result.reason,
                            ack_ts=int(exec_result.ack_ts),
                            seen_to_ack_ms=int(exec_result.seen_to_ack_ms),
                        )
                        if not exec_result.ok:
                            fail_reason = exec_result.reason or "LIVE_POST_FAILED"
                            skip_reasons[fail_reason] = skip_reasons.get(fail_reason, 0) + 1
                        else:
                            _apply_confirmed_live_fill(exec_shadow=shadow_exec_live, base_action=action_l, exec_result=exec_result)
                        action = action_l
                    elif reason is None and live and live_exec is not None:
                        if side == "BUY":
                            if bool(taker or cfg.live_taker):
                                live_price = float(exec_price)
                            else:
                                live_price = _quantize_down(float(exec_price) - float(mapping.tick_size), float(mapping.tick_size))
                                live_price = max(float(mapping.tick_size), float(live_price))
                        else:
                            if bool(taker or cfg.live_taker):
                                live_price = float(exec_price)
                            else:
                                live_price = _quantize_up(float(exec_price) + float(mapping.tick_size), float(mapping.tick_size))
                        action = Action(
                            run_id=int(start_ts),
                            seen_ts=int(seen_ts),
                            trade_ts=int(mt.ts_last),
                            condition_id=mt.condition_id,
                            outcome_index=int(mt.outcome_index),
                            token_id=str(mapping.token_id),
                            side=side,
                            usdc=float(you_cost),
                            shares=float(shares),
                            price=float(live_price),
                            tick_size=float(mapping.tick_size),
                            neg_risk=bool(mapping.neg_risk),
                        )
                        live_order_attempt_ts.append(int(time.time()))
                        if side == "BUY":
                            day_key = _now_utc_date()
                            live_buy_budget_by_utc_day[day_key] = float(live_buy_budget_by_utc_day.get(day_key, 0.0)) + float(you_cost)
                        exec_result = live_exec.execute(action)
                        db.add_live_order_timing(
                            run_id=int(start_ts),
                            ts=int(decision_ts),
                            seen_ts=int(seen_ts),
                            trade_ts=int(mt.ts_last),
                            staleness_sec=float(int(seen_ts) - int(mt.ts_last)),
                            side=str(side),
                            condition_id=str(mt.condition_id),
                            token_id=str(mapping.token_id),
                            usdc=float(you_cost),
                            shares=float(shares),
                            price=float(live_price),
                            ok=bool(exec_result.ok),
                            order_id=exec_result.order_id,
                            reason=exec_result.reason,
                            ack_ts=int(exec_result.ack_ts),
                            seen_to_ack_ms=int(exec_result.seen_to_ack_ms),
                        )
                        if not exec_result.ok:
                            fail_reason = exec_result.reason or "LIVE_POST_FAILED"
                            skip_reasons[fail_reason] = skip_reasons.get(fail_reason, 0) + 1
                        else:
                            _apply_confirmed_live_fill(exec_shadow=shadow_exec, base_action=action, exec_result=exec_result)
                    else:
                        # shadow 执行：写入影子账本
                        action = Action(
                            run_id=int(start_ts),
                            seen_ts=int(seen_ts),
                            trade_ts=int(mt.ts_last),
                            condition_id=mt.condition_id,
                            outcome_index=int(mt.outcome_index),
                            token_id=str(mapping.token_id),
                            side=side,
                            usdc=float(you_cost),
                            shares=float(shares),
                            price=float(exec_price),
                            tick_size=float(mapping.tick_size),
                            neg_risk=bool(mapping.neg_risk),
                        )
                        exec_result = shadow_exec.execute(action)

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
                        "quote": {"best_ask": best_ask, "best_bid": best_bid, "exec_price": exec_price, "slip_abs": slip_abs},
                        "you": {"his_cost_usdc": his_cost, "you_cost_usdc": you_cost, "k": cfg.shadow_k},
                        "action": action.__dict__ if action else None,
                        "exec": exec_result.__dict__ if exec_result else None,
                        "ledger": db.get_ledger_state().__dict__,
                    },
                )

                # KPI 增量写（cheap）
                if int(time.time()) - last_kpi_write_at >= 10:
                    if dual and db_live is not None:
                        kpi_s = _build_kpi(db, "SHADOW", None)
                        kpi_l = _build_kpi(db_live, "LIVE", db)
                        kpi_shadow_json.write_text(json.dumps(kpi_s, ensure_ascii=False, indent=2), encoding="utf-8")
                        kpi_shadow_md.write_text(_render_kpi_md(kpi=kpi_s), encoding="utf-8")
                        kpi_live_json.write_text(json.dumps(kpi_l, ensure_ascii=False, indent=2), encoding="utf-8")
                        kpi_live_md.write_text(_render_kpi_md(kpi=kpi_l), encoding="utf-8")
                    else:
                        mode_str = "LIVE" if live else "SHADOW"
                        kpi = _build_kpi(db, mode_str, None)
                        kpi_json_path.write_text(json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")
                        kpi_md_path.write_text(_render_kpi_md(kpi=kpi), encoding="utf-8")
                    last_kpi_write_at = int(time.time())

            time.sleep(max(0.5, cfg.shadow_poll_interval_sec + random.uniform(-cfg.shadow_jitter_sec, cfg.shadow_jitter_sec)))

        # 结束时写一次最终 KPI（即使中途没触发增量写）
        if dual and db_live is not None:
            kpi_s = _build_kpi(db, "SHADOW", None)
            kpi_l = _build_kpi(db_live, "LIVE", db)
            kpi_shadow_json.write_text(json.dumps(kpi_s, ensure_ascii=False, indent=2), encoding="utf-8")
            kpi_shadow_md.write_text(_render_kpi_md(kpi=kpi_s), encoding="utf-8")
            kpi_live_json.write_text(json.dumps(kpi_l, ensure_ascii=False, indent=2), encoding="utf-8")
            kpi_live_md.write_text(_render_kpi_md(kpi=kpi_l), encoding="utf-8")
            _write_jsonl(log_path, {"type": "end", "ts": int(time.time()), "kpi_shadow": kpi_s, "kpi_live": kpi_l})
        else:
            mode_str = "LIVE" if live else "SHADOW"
            kpi = _build_kpi(db, mode_str, None)
            kpi_json_path.write_text(json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")
            kpi_md_path.write_text(_render_kpi_md(kpi=kpi), encoding="utf-8")
            _write_jsonl(log_path, {"type": "end", "ts": int(time.time()), "kpi": kpi})
        return 0
    finally:
        api.close()
        clob.close()
        if live_exec is not None:
            live_exec.close()

