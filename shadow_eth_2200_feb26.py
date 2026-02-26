from __future__ import annotations

"""
影子下单示例（不下单、不需要私钥）：

目标：买入 “Will the price of Ethereum be above $2,200 on February 26?” 的 YES，
按“预算 USDC -> 反推 shares”的方式生成一条纸面订单计划，并打印。

运行（在项目目录）：
  python shadow_eth_2200_feb26.py --usdc 1
"""

import argparse
import json
from dataclasses import dataclass
from typing import Any

import httpx


GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

EVENT_SLUG = "ethereum-above-on-february-26"
MARKET_SLUG = "ethereum-above-2200-on-february-26"


@dataclass(frozen=True)
class ShadowOrderPlan:
    event_slug: str
    market_slug: str
    question: str
    condition_id: str
    token_id_yes: str
    side: str  # BUY
    usdc_budget: float
    limit_price: float
    est_shares: float
    min_order_size: float
    best_buy_price: float
    best_sell_price: float
    notes: list[str]


def _get_json(url: str, *, params: dict[str, Any] | None = None) -> Any:
    r = httpx.get(
        url,
        params=params,
        timeout=20.0,
        headers={"accept": "application/json", "user-agent": "pm-shadow/0.1"},
    )
    r.raise_for_status()
    return r.json()


def fetch_market_condition_id() -> tuple[str, str]:
    events = _get_json(f"{GAMMA}/events", params={"slug": EVENT_SLUG})
    if not events:
        raise RuntimeError(f"Gamma events?slug={EVENT_SLUG} 返回空结果")
    ev = events[0]
    markets = ev.get("markets") or []
    m = next((x for x in markets if x.get("slug") == MARKET_SLUG), None)
    if not m:
        got = [x.get("slug") for x in markets][:20]
        raise RuntimeError(f"找不到 market_slug={MARKET_SLUG}，示例 slugs={got}")
    return str(m["conditionId"]), str(m.get("question") or m.get("title") or "")


def fetch_yes_token_id(condition_id: str) -> tuple[str, float]:
    m = _get_json(f"{CLOB}/markets/{condition_id}")
    tokens = m.get("tokens") or []
    yes = next((t for t in tokens if str(t.get("outcome", "")).lower() == "yes"), None)
    if not yes:
        raise RuntimeError(f"CLOB market 没找到 YES token，tokens={tokens}")
    token_id = str(yes["token_id"])
    min_order_size = float(m.get("minimum_order_size") or 0.0)
    return token_id, min_order_size


def fetch_best_prices(token_id: str) -> tuple[float, float]:
    # /price: side=BUY 给出“买入该 token 的最佳价”（通常可视为 best ask）
    buy = _get_json(f"{CLOB}/price", params={"token_id": token_id, "side": "BUY"})
    sell = _get_json(f"{CLOB}/price", params={"token_id": token_id, "side": "SELL"})
    return float(buy["price"]), float(sell["price"])


def build_shadow_plan(*, usdc: float) -> ShadowOrderPlan:
    condition_id, question = fetch_market_condition_id()
    token_id_yes, min_order_size = fetch_yes_token_id(condition_id)
    best_buy, best_sell = fetch_best_prices(token_id_yes)

    # 影子计划：用 best_buy 作为“你此刻要成交需要付出的价格”的参考
    limit_price = best_buy
    est_shares = usdc / max(1e-9, limit_price)

    notes: list[str] = []
    if min_order_size and est_shares < min_order_size:
        notes.append(f"WARNING: est_shares({est_shares:.6f}) < minimum_order_size({min_order_size})，实盘可能会被拒单")
    notes.append("Shadow-only: this script DOES NOT place any real orders.")
    notes.append("If you want a more conservative paper quote, set limit_price below best_buy (post-only style), but it may not fill.")

    return ShadowOrderPlan(
        event_slug=EVENT_SLUG,
        market_slug=MARKET_SLUG,
        question=question,
        condition_id=condition_id,
        token_id_yes=token_id_yes,
        side="BUY",
        usdc_budget=float(usdc),
        limit_price=float(limit_price),
        est_shares=float(est_shares),
        min_order_size=float(min_order_size),
        best_buy_price=float(best_buy),
        best_sell_price=float(best_sell),
        notes=notes,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usdc", type=float, default=1.0, help="纸面下单预算（USDC），默认 1")
    args = ap.parse_args()

    plan = build_shadow_plan(usdc=float(args.usdc))
    print(json.dumps(plan.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

