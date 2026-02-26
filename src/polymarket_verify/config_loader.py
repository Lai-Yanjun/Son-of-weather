from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    username: str
    address: str

    data_api_base: str
    timeout_sec: float
    activity_max_trades: int
    closed_positions_max: int

    same_second_cluster_gap_sec: int
    suggested_merge_window_sec: int
    round_trip_window_sec: int

    mdd_budget_usdc: float
    max_total_cost_exposure_usdc: float

    poll_interval_sec: float
    jitter_sec: float

    shadow_initial_cash_usdc: float
    shadow_follow_all: bool
    shadow_poll_interval_sec: float
    shadow_jitter_sec: float
    shadow_fetch_limit: int
    shadow_merge_window_sec: int
    shadow_dedupe_ttl_sec: int
    shadow_k: float
    shadow_min_per_trade_usdc: float
    shadow_max_per_trade_usdc: float
    shadow_max_market_exposure_usdc: float
    shadow_max_total_exposure_usdc: float
    shadow_daily_loss_limit_usdc: float
    shadow_max_abs_slippage: float
    shadow_condition_whitelist: list[str]


def _get(d: dict[str, Any], key: str, default: Any = None) -> Any:
    return d[key] if key in d else default


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config.yaml 必须是一个 YAML mapping")

    target = raw.get("target", {}) or {}
    api = raw.get("api", {}) or {}
    analysis = raw.get("analysis", {}) or {}
    risk = raw.get("risk_budget_example", {}) or {}
    polling = raw.get("polling_suggestion", {}) or {}
    shadow = raw.get("shadow", {}) or {}

    return AppConfig(
        username=str(target["username"]),
        address=str(target["address"]).lower(),
        data_api_base=str(_get(api, "data_api_base", "https://data-api.polymarket.com")).rstrip("/"),
        timeout_sec=float(_get(api, "timeout_sec", 20)),
        activity_max_trades=int(_get(api, "activity_max_trades", 2000)),
        closed_positions_max=int(_get(api, "closed_positions_max", 3000)),
        same_second_cluster_gap_sec=int(_get(analysis, "same_second_cluster_gap_sec", 1)),
        suggested_merge_window_sec=int(_get(analysis, "suggested_merge_window_sec", 2)),
        round_trip_window_sec=int(_get(analysis, "round_trip_window_sec", 600)),
        mdd_budget_usdc=float(_get(risk, "mdd_budget_usdc", 50)),
        max_total_cost_exposure_usdc=float(_get(risk, "max_total_cost_exposure_usdc", 100)),
        poll_interval_sec=float(_get(polling, "poll_interval_sec", 3)),
        jitter_sec=float(_get(polling, "jitter_sec", 0.5)),

        shadow_initial_cash_usdc=float(_get(shadow, "initial_cash_usdc", 400)),
        shadow_follow_all=bool(_get(shadow, "follow_all", False)),
        shadow_poll_interval_sec=float(_get(shadow, "poll_interval_sec", 5)),
        shadow_jitter_sec=float(_get(shadow, "jitter_sec", 0.8)),
        shadow_fetch_limit=int(_get(shadow, "fetch_limit", 120)),
        shadow_merge_window_sec=int(_get(shadow, "merge_window_sec", 2)),
        shadow_dedupe_ttl_sec=int(_get(shadow, "dedupe_ttl_sec", 1209600)),
        shadow_k=float(_get(shadow, "k", 0.25)),
        shadow_min_per_trade_usdc=float(_get(shadow, "min_per_trade_usdc", 5)),
        shadow_max_per_trade_usdc=float(_get(shadow, "max_per_trade_usdc", 25)),
        shadow_max_market_exposure_usdc=float(_get(shadow, "max_market_exposure_usdc", 120)),
        shadow_max_total_exposure_usdc=float(_get(shadow, "max_total_exposure_usdc", 240)),
        shadow_daily_loss_limit_usdc=float(_get(shadow, "daily_loss_limit_usdc", 60)),
        shadow_max_abs_slippage=float(_get(shadow, "max_abs_slippage", 0.015)),
        shadow_condition_whitelist=[str(x) for x in (_get(shadow, "condition_whitelist", []) or [])],
    )

