from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import execution_profile, market_profile, mdd_and_k
from .config_loader import load_config
from .data_api import DataApiClient
from .playbook import make_parameter_playbook
from .polling import polling_and_realtime_advice
from .report_md import render_report_md


def cmd_report(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = DataApiClient(base_url=cfg.data_api_base, timeout_sec=cfg.timeout_sec)
    try:
        trades = list(api.iter_activity_trades(user=cfg.address, max_trades=cfg.activity_max_trades))
        positions = api.get_positions(user=cfg.address)
        closed_positions = list(api.iter_closed_positions(user=cfg.address, max_items=cfg.closed_positions_max))
    finally:
        api.close()

    min_ts = min((t.timestamp for t in trades), default=None)
    max_ts = max((t.timestamp for t in trades), default=None)

    mp = market_profile(trades)
    ep = execution_profile(
        trades,
        same_second_cluster_gap_sec=cfg.same_second_cluster_gap_sec,
        round_trip_window_sec=cfg.round_trip_window_sec,
    )

    mk = mdd_and_k(
        closed_positions,
        positions,
        mdd_budget_usdc=cfg.mdd_budget_usdc,
        max_total_cost_exposure_usdc=cfg.max_total_cost_exposure_usdc,
    )

    polling = polling_and_realtime_advice(
        ep,
        suggested_poll_interval_sec=cfg.poll_interval_sec,
        jitter_sec=cfg.jitter_sec,
    )

    stats = {
        "username": cfg.username,
        "address": cfg.address,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "market_profile": mp,
        "execution_profile": ep,
        "polling_and_realtime": polling,
        "mdd_and_k": mk,
    }
    stats["parameter_playbook"] = make_parameter_playbook(stats=stats)

    report_md = render_report_md(username=cfg.username, address=cfg.address, stats=stats)

    report_path = out_dir / f"{cfg.username}_report.md"
    json_path = out_dir / f"{cfg.username}_stats.json"
    report_path.write_text(report_md, encoding="utf-8")
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(report_path))
    print(str(json_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="polymarket-verify", description="Polymarket 跟单：阶段0验证与参数校准工具")
    p.add_argument("--config", default="config.yaml", help="配置文件路径（默认 config.yaml）")
    p.add_argument("--out-dir", default="reports", help="输出目录（默认 reports/）")

    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("report", help="拉取数据并生成验证报告（markdown + json）")
    sp.set_defaults(func=cmd_report)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

