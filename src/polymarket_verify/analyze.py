from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .models import ActivityTrade, ClosedPosition, Position
from .stats import max_drawdown, summarize


@dataclass(frozen=True)
class MarketAgg:
    conditionId: str
    eventSlug: str
    slug: str
    title: str
    trades: int
    buys: int
    sells: int
    usdc_total: float
    usdc_buy: float
    usdc_sell: float
    first_ts: int
    last_ts: int


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def market_profile(trades: list[ActivityTrade]) -> dict[str, Any]:
    by_condition: dict[str, list[ActivityTrade]] = defaultdict(list)
    for t in trades:
        by_condition[t.conditionId].append(t)

    aggs: list[MarketAgg] = []
    for cid, ts in by_condition.items():
        ts_sorted = sorted(ts, key=lambda x: x.timestamp)
        buys = sum(1 for x in ts_sorted if x.side == "BUY")
        sells = len(ts_sorted) - buys
        usdc_total = float(sum(x.usdcSize for x in ts_sorted))
        usdc_buy = float(sum(x.usdcSize for x in ts_sorted if x.side == "BUY"))
        usdc_sell = float(sum(x.usdcSize for x in ts_sorted if x.side == "SELL"))
        sample = ts_sorted[-1]
        aggs.append(
            MarketAgg(
                conditionId=cid,
                eventSlug=_safe_str(sample.eventSlug),
                slug=_safe_str(sample.slug),
                title=_safe_str(sample.title),
                trades=len(ts_sorted),
                buys=buys,
                sells=sells,
                usdc_total=usdc_total,
                usdc_buy=usdc_buy,
                usdc_sell=usdc_sell,
                first_ts=ts_sorted[0].timestamp,
                last_ts=ts_sorted[-1].timestamp,
            )
        )

    aggs.sort(key=lambda a: (a.usdc_total, a.trades), reverse=True)
    total_trades = len(trades)
    total_usdc = float(sum(t.usdcSize for t in trades))

    def share_top(n: int) -> dict[str, float]:
        top = aggs[:n]
        return {
            "top_n": n,
            "trade_share": float(sum(x.trades for x in top) / max(1, total_trades)),
            "usdc_share": float(sum(x.usdc_total for x in top) / max(1e-9, total_usdc)),
        }

    event_counter = Counter(_safe_str(t.eventSlug) for t in trades)
    return {
        "total_trades": total_trades,
        "total_usdc": total_usdc,
        "unique_conditions": len(by_condition),
        "unique_events": len([k for k, v in event_counter.items() if k]),
        "concentration": {
            "top3": share_top(3),
            "top5": share_top(5),
            "top10": share_top(10),
        },
        "top_markets": [
            {
                "conditionId": a.conditionId,
                "eventSlug": a.eventSlug,
                "slug": a.slug,
                "title": a.title,
                "trades": a.trades,
                "buys": a.buys,
                "sells": a.sells,
                "usdc_total": round(a.usdc_total, 6),
                "first_ts": a.first_ts,
                "last_ts": a.last_ts,
            }
            for a in aggs[:25]
        ],
        "top_events": [
            {"eventSlug": k, "trades": int(v)}
            for k, v in event_counter.most_common(25)
            if k
        ],
    }


def execution_profile(
    trades: list[ActivityTrade],
    *,
    same_second_cluster_gap_sec: int = 1,
    round_trip_window_sec: int = 600,
) -> dict[str, Any]:
    if not trades:
        return {}

    # 按时间升序做间隔统计
    ts_sorted = sorted(trades, key=lambda x: (x.timestamp, x.transactionHash, x.asset))
    deltas = [ts_sorted[i].timestamp - ts_sorted[i - 1].timestamp for i in range(1, len(ts_sorted))]

    # 同秒/近同秒连打簇（不区分市场）
    clusters: list[int] = []
    cur = 1
    for d in deltas:
        if d <= same_second_cluster_gap_sec:
            cur += 1
        else:
            clusters.append(cur)
            cur = 1
    clusters.append(cur)

    # 每秒成交笔数分布（用于判断“爆发式连打”）
    per_sec = Counter(t.timestamp for t in ts_sorted)
    per_sec_counts = list(per_sec.values())
    per_sec_counts.sort(reverse=True)

    # 单笔规模（USDC）
    usdc_sizes = [float(t.usdcSize) for t in ts_sorted]
    size_tokens = [float(t.size) for t in ts_sorted]

    side_counts = Counter(t.side for t in ts_sorted)

    # 来回手（同 asset，反向成交在 N 秒内出现）
    # 近似：对每个 asset 维护上一次 BUY/SELL 的时间戳，出现反向且在窗口内则计数
    last_by_asset_side: dict[tuple[str, str], int] = {}
    round_trip_hits = 0
    for t in ts_sorted:
        other_side = "SELL" if t.side == "BUY" else "BUY"
        key_other = (t.asset, other_side)
        if key_other in last_by_asset_side:
            if t.timestamp - last_by_asset_side[key_other] <= round_trip_window_sec:
                round_trip_hits += 1
        last_by_asset_side[(t.asset, t.side)] = t.timestamp

    return {
        "time_span_sec": int(ts_sorted[-1].timestamp - ts_sorted[0].timestamp),
        "deltas_sec": summarize(deltas),
        "clusters": {
            "gap_sec": same_second_cluster_gap_sec,
            "cluster_size_summary": summarize(clusters),
            "p_cluster_ge_2": float(sum(1 for x in clusters if x >= 2) / max(1, len(clusters))),
            "max_cluster_size": int(max(clusters) if clusters else 1),
        },
        "per_second": {
            "max_trades_in_one_second": int(per_sec_counts[0]) if per_sec_counts else 0,
            "p_seconds_ge_2": float(sum(1 for x in per_sec_counts if x >= 2) / max(1, len(per_sec_counts))),
            "top10_seconds_trade_counts": [int(x) for x in per_sec_counts[:10]],
        },
        "usdc_size": summarize(usdc_sizes),
        "token_size": summarize(size_tokens),
        "side_ratio": {
            "BUY": int(side_counts.get("BUY", 0)),
            "SELL": int(side_counts.get("SELL", 0)),
            "buy_share": float(side_counts.get("BUY", 0) / max(1, len(ts_sorted))),
        },
        "round_trip": {
            "window_sec": int(round_trip_window_sec),
            "hits": int(round_trip_hits),
            "hit_rate_per_trade": float(round_trip_hits / max(1, len(ts_sorted))),
        },
    }


def mdd_and_k(
    closed_positions: list[ClosedPosition],
    positions: list[Position],
    *,
    mdd_budget_usdc: float,
    max_total_cost_exposure_usdc: float,
) -> dict[str, Any]:
    # 已实现权益曲线（每个 closed position 视为一次“结算现金流”）
    cps = sorted(closed_positions, key=lambda x: x.timestamp)
    realized_steps = [float(cp.realizedPnl) for cp in cps]
    equity = []
    cur = 0.0
    for x in realized_steps:
        cur += x
        equity.append(cur)

    realized_mdd = max_drawdown(equity, baseline=0.0)
    realized_now = float(equity[-1] if equity else 0.0)

    # 当前开放仓位的未实现 PnL
    unrealized_sum = float(sum(p.cashPnl for p in positions))
    # 由于 closed-positions 的样本可能很小（且可能偏向“赢家样本”），我们用更保守的代理：
    # 1) 当前所有“浮亏仓位”的亏损总和（忽略浮盈）——这给出一个“当下风险/回撤”尺度；
    # 2) 已实现权益曲线的 MDD（若存在亏损闭仓，会反映出来）。
    open_drawdown_abs = float(sum(max(0.0, -float(p.cashPnl)) for p in positions))

    mdd_proxy = max(float(realized_mdd.mdd_abs), float(open_drawdown_abs))

    # 成本敞口（用 initialValue 累加，等价于 size*avgPrice）
    cost_exposure = float(sum(p.initialValue for p in positions))
    # 市值（用 currentValue 累加）
    market_value = float(sum(p.currentValue for p in positions))

    # 给出建议 k（若 MDD_his 为 0，则该约束不生效）
    if mdd_proxy <= 1e-9:
        k_by_mdd = None
    else:
        k_by_mdd = float(mdd_budget_usdc / mdd_proxy)
    if cost_exposure <= 1e-9:
        k_by_cost = None
    else:
        k_by_cost = float(max_total_cost_exposure_usdc / cost_exposure)

    k_suggest = None
    candidates = [x for x in [k_by_mdd, k_by_cost] if x is not None]
    if candidates:
        k_suggest = float(min(candidates))

    # 持仓集中度
    pos_sorted = sorted(positions, key=lambda p: p.initialValue, reverse=True)
    top = []
    for p in pos_sorted[:15]:
        top.append(
            {
                "conditionId": p.conditionId,
                "slug": p.slug,
                "title": p.title,
                "outcomeIndex": p.outcomeIndex,
                "initialValue": round(float(p.initialValue), 6),
                "currentValue": round(float(p.currentValue), 6),
                "cashPnl": round(float(p.cashPnl), 6),
                "curPrice": float(p.curPrice),
                "avgPrice": float(p.avgPrice),
            }
        )

    def top_share(n: int) -> float:
        return float(sum(p.initialValue for p in pos_sorted[:n]) / max(1e-9, cost_exposure))

    return {
        "closed_positions_count": int(len(closed_positions)),
        "realized_pnl_sum": round(realized_now, 6),
        "unrealized_cash_pnl_now": round(unrealized_sum, 6),
        "open_drawdown_abs_now": round(open_drawdown_abs, 6),
        "realized_mdd_abs": round(realized_mdd.mdd_abs, 6),
        "mdd_proxy_abs": round(mdd_proxy, 6),
        "cost_exposure_usdc": round(cost_exposure, 6),
        "market_value_usdc": round(market_value, 6),
        "k_constraints": {
            "mdd_budget_usdc": float(mdd_budget_usdc),
            "max_total_cost_exposure_usdc": float(max_total_cost_exposure_usdc),
            "k_by_mdd": None if k_by_mdd is None else round(k_by_mdd, 6),
            "k_by_cost": None if k_by_cost is None else round(k_by_cost, 6),
            "k_suggest": None if k_suggest is None else round(k_suggest, 6),
        },
        "concentration": {
            "top1_cost_share": round(top_share(1), 6) if pos_sorted else None,
            "top3_cost_share": round(top_share(3), 6) if pos_sorted else None,
            "top5_cost_share": round(top_share(5), 6) if pos_sorted else None,
        },
        "top_positions": top,
    }

