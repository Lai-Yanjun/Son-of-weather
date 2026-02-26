from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _ts(ts: int | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_report_md(*, username: str, address: str, stats: dict[str, Any]) -> str:
    mp = stats.get("market_profile", {})
    ep = stats.get("execution_profile", {})
    mk = stats.get("mdd_and_k", {})
    polling = stats.get("polling_and_realtime", {})

    span_sec = ep.get("time_span_sec") or 0
    hours = span_sec / 3600 if span_sec else None
    tph = (mp.get("total_trades") / hours) if hours else None

    lines: list[str] = []
    lines.append(f"## neobrother 验证报告（阶段0）")
    lines.append("")
    lines.append(f"- **user**: `{username}`")
    lines.append(f"- **address**: `{address}`")
    lines.append("")

    lines.append("## 1) 市场画像（做的是什么市场）")
    lines.append("")
    lines.append(f"- **统计样本**: trades={mp.get('total_trades')}，unique_conditions={mp.get('unique_conditions')}，总成交额(usdcSize 累加)≈{_fmt(mp.get('total_usdc'), 6)} USDC")
    if hours:
        lines.append(f"- **时间跨度**: ≈{_fmt(hours, 2)} 小时（{_ts(stats.get('min_ts'))} → {_ts(stats.get('max_ts'))}）")
        lines.append(f"- **频率**: ≈{_fmt(tph, 2)} trades/hour")
    lines.append("")
    conc = mp.get("concentration", {})
    if conc:
        lines.append(f"- **集中度（按 USDC / 笔数）**: top3={_fmt(conc.get('top3', {}).get('usdc_share'))}/{_fmt(conc.get('top3', {}).get('trade_share'))}，top5={_fmt(conc.get('top5', {}).get('usdc_share'))}/{_fmt(conc.get('top5', {}).get('trade_share'))}")
        lines.append("")

    lines.append("### Top markets（按 USDC）")
    lines.append("")
    lines.append("|rank|usdc_total|trades|buy/sell|title|slug|")
    lines.append("|---:|---:|---:|---:|---|---|")
    for i, m in enumerate((mp.get("top_markets") or [])[:15], start=1):
        lines.append(
            f"|{i}|{_fmt(m.get('usdc_total'), 6)}|{m.get('trades')}|{m.get('buys')}/{m.get('sells')}|{_safe_md(m.get('title'))}|{_safe_md(m.get('slug'))}|"
        )
    lines.append("")

    lines.append("### Top eventSlug（按 trades）")
    lines.append("")
    lines.append("|rank|trades|eventSlug|")
    lines.append("|---:|---:|---|")
    for i, e in enumerate((mp.get("top_events") or [])[:10], start=1):
        lines.append(f"|{i}|{e.get('trades')}|{_safe_md(e.get('eventSlug'))}|")
    lines.append("")

    lines.append("## 2) 成交/订单特征（是否拆单、同秒连打、来回手）")
    lines.append("")
    lines.append(f"- **BUY 占比**: {_fmt(ep.get('side_ratio', {}).get('buy_share'))}（BUY={ep.get('side_ratio', {}).get('BUY')} / SELL={ep.get('side_ratio', {}).get('SELL')}）")
    lines.append(f"- **单笔 USDC（usdcSize）**: p50={_fmt(ep.get('usdc_size', {}).get('p50'), 6)}，p90={_fmt(ep.get('usdc_size', {}).get('p90'), 6)}，max={_fmt(ep.get('usdc_size', {}).get('max'), 6)}")
    lines.append("")
    lines.append(f"- **相邻成交间隔（秒）**: p50={_fmt(ep.get('deltas_sec', {}).get('p50'))}，p90={_fmt(ep.get('deltas_sec', {}).get('p90'))}，min={_fmt(ep.get('deltas_sec', {}).get('min'))}")
    lines.append(f"- **同秒/近同秒连打簇**(gap≤{ep.get('clusters', {}).get('gap_sec')}s): p(cluster≥2)={_fmt(ep.get('clusters', {}).get('p_cluster_ge_2'))}，max_cluster={ep.get('clusters', {}).get('max_cluster_size')}")
    lines.append(f"- **每秒成交爆发**: max_trades_in_1s={ep.get('per_second', {}).get('max_trades_in_one_second')}，p(seconds≥2)={_fmt(ep.get('per_second', {}).get('p_seconds_ge_2'))}")
    lines.append("")
    lines.append(f"- **来回手(近似)**: window={ep.get('round_trip', {}).get('window_sec')}s，hit_rate/trade={_fmt(ep.get('round_trip', {}).get('hit_rate_per_trade'))}")
    lines.append("")

    lines.append("## 3) 延迟是否关键？Data API 能多实时？轮询间隔怎么定？")
    lines.append("")
    lines.append(polling.get("conclusion", "- -"))
    lines.append("")
    lines.append("### 建议（工程化可执行）")
    lines.append("")
    for b in polling.get("bullets", []):
        lines.append(f"- {b}")
    lines.append("")

    lines.append("## 4) 最大回撤（近似）与跟单比例 k 怎么选")
    lines.append("")
    lines.append(f"- **closed-positions 样本数**: {mk.get('closed_positions_count')}")
    lines.append(f"- **已实现 PnL 累计（近似）**: {_fmt(mk.get('realized_pnl_sum'), 6)} USDC")
    lines.append(f"- **当前未实现 PnL（positions.cashPnl 合计）**: {_fmt(mk.get('unrealized_cash_pnl_now'), 6)} USDC")
    lines.append(f"- **MDD（已实现权益曲线，近似）**: {_fmt(mk.get('realized_mdd_abs'), 6)} USDC")
    lines.append(f"- **当前开放仓位浮亏合计（忽略浮盈，回撤代理）**: {_fmt(mk.get('open_drawdown_abs_now'), 6)} USDC")
    lines.append(f"- **MDD_proxy（用于保守 k 约束）**: {_fmt(mk.get('mdd_proxy_abs'), 6)} USDC")
    lines.append("")
    lines.append(f"- **当前成本敞口 C_his（initialValue 合计）**: {_fmt(mk.get('cost_exposure_usdc'), 6)} USDC")
    lines.append(f"- **当前市值 V_his（currentValue 合计）**: {_fmt(mk.get('market_value_usdc'), 6)} USDC")
    lines.append("")
    kc = mk.get("k_constraints", {})
    lines.append("### k 建议（按你在 config 里给的预算示例）")
    lines.append("")
    lines.append(f"- **预算**: MDD_budget={_fmt(kc.get('mdd_budget_usdc'))}，max_total_cost_exposure={_fmt(kc.get('max_total_cost_exposure_usdc'))}")
    lines.append(f"- **k_by_mdd**: {_fmt(kc.get('k_by_mdd'), 6)}")
    lines.append(f"- **k_by_cost**: {_fmt(kc.get('k_by_cost'), 6)}")
    lines.append(f"- **k_suggest = min(k_by_mdd, k_by_cost)**: {_fmt(kc.get('k_suggest'), 6)}")
    lines.append("")

    lines.append("### 持仓集中度（用于设置单市场/总敞口上限）")
    lines.append("")
    conc2 = mk.get("concentration", {})
    lines.append(f"- **top1/top3/top5 成本集中度**: {conc2.get('top1_cost_share')} / {conc2.get('top3_cost_share')} / {conc2.get('top5_cost_share')}")
    lines.append("")

    lines.append("## 5) 参数怎么调（从验证结论到可执行参数）")
    lines.append("")
    for b in stats.get("parameter_playbook", []):
        lines.append(f"- {b}")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _safe_md(s: Any) -> str:
    if s is None:
        return ""
    txt = str(s)
    return txt.replace("|", "\\|").replace("\n", " ").strip()

