from __future__ import annotations

"""
实盘交易测试（极小额）——ETH > 2200 (Feb 26) 买 YES

重要：
- 默认只打印计划（dry-run），不会下单。
- 只有显式传 --live 才会真的发单。
- 请确保你所在地区与账户使用符合 Polymarket 条款与限制。

准备：
1) 复制 .env.example -> .env 并填写 PRIVATE_KEY / SIGNATURE_TYPE / FUNDER_ADDRESS
2) pip install -r requirements.txt

运行（在项目目录）：
  # dry-run（推荐先跑）
  python live_trade_test_eth_2200_feb26_yes.py --usdc 1

  # 真下单（小额测试）
  python live_trade_test_eth_2200_feb26_yes.py --usdc 1 --live

说明：
- 下单使用 GTC 限价单，默认 post-only（不吃单，若会立即成交则被拒单）。
- 你可以加 --taker 让它变成“可成交”来测试撮合，但那会真的成交并产生风险/费用。
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, MarketOrderArgs, OrderArgs, OrderType, PartialCreateOrderOptions
from py_clob_client.order_builder.constants import BUY


GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet

EVENT_SLUG = "ethereum-above-on-february-26"
MARKET_SLUG = "ethereum-above-2200-on-february-26"


@dataclass(frozen=True)
class ResolvedMarket:
    question: str
    condition_id: str
    token_id_yes: str
    min_order_size: float
    tick_size: float
    neg_risk: bool


def _get_json(url: str, *, params: dict[str, Any] | None = None) -> Any:
    r = httpx.get(
        url,
        params=params,
        timeout=20.0,
        headers={"accept": "application/json", "user-agent": "pm-live-test/0.1"},
    )
    r.raise_for_status()
    return r.json()


def resolve_eth_2200_yes_market() -> ResolvedMarket:
    events = _get_json(f"{GAMMA}/events", params={"slug": EVENT_SLUG})
    if not events:
        raise RuntimeError(f"Gamma events?slug={EVENT_SLUG} 返回空结果")
    ev = events[0]
    markets = ev.get("markets") or []
    m = next((x for x in markets if x.get("slug") == MARKET_SLUG), None)
    if not m:
        got = [x.get("slug") for x in markets][:20]
        raise RuntimeError(f"找不到 market_slug={MARKET_SLUG}，示例 slugs={got}")

    condition_id = str(m["conditionId"])
    question = str(m.get("question") or "")

    clob_market = _get_json(f"{CLOB}/markets/{condition_id}")
    tokens = clob_market.get("tokens") or []
    yes = next((t for t in tokens if str(t.get("outcome", "")).lower() == "yes"), None)
    if not yes:
        raise RuntimeError(f"CLOB market 没找到 YES token，tokens={tokens}")

    token_id_yes = str(yes["token_id"])
    min_order_size = float(clob_market.get("minimum_order_size") or 0.0)
    tick_size = float(clob_market.get("minimum_tick_size") or 0.01)
    neg_risk = bool(clob_market.get("neg_risk") or clob_market.get("negative_risk") or False)

    return ResolvedMarket(
        question=question,
        condition_id=condition_id,
        token_id_yes=token_id_yes,
        min_order_size=min_order_size,
        tick_size=tick_size,
        neg_risk=neg_risk,
    )


def get_best_bid_ask(token_id: str) -> tuple[Optional[float], Optional[float]]:
    book = _get_json(f"{CLOB}/book", params={"token_id": token_id})
    bids = [float(x["price"]) for x in (book.get("bids") or []) if "price" in x]
    asks = [float(x["price"]) for x in (book.get("asks") or []) if "price" in x]
    best_bid = max(bids) if bids else None
    best_ask = min(asks) if asks else None
    return best_bid, best_ask


def quantize_down(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    n = int(price / tick + 1e-12)
    return n * tick


def quantize_up(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    n = int(price / tick + 1 - 1e-12)
    return n * tick


def tick_size_to_literal(tick: float) -> str:
    """
    py-clob-client 的 tick_size 只接受: 0.1 / 0.01 / 0.001 / 0.0001（字符串）。
    """
    mapping = {
        0.1: "0.1",
        0.01: "0.01",
        0.001: "0.001",
        0.0001: "0.0001",
    }
    # 允许浮点误差
    for k, v in mapping.items():
        if abs(float(tick) - k) <= 1e-9:
            return v
    # fallback：尽量格式化成文档允许的样式（若不在集合内，下单会被拒）
    s = f"{tick:.4f}".rstrip("0").rstrip(".")
    return s


def build_client(*, host: str, chain_id: int) -> ClobClient:
    pk = os.getenv("PRIVATE_KEY")
    if not pk:
        raise RuntimeError("缺少 PRIVATE_KEY（请在 .env 填写）")

    sig_type = int(os.getenv("SIGNATURE_TYPE", "0"))
    funder = os.getenv("FUNDER_ADDRESS") or None

    api_key = os.getenv("POLY_API_KEY") or None
    api_secret = os.getenv("POLY_SECRET") or None
    api_pass = os.getenv("POLY_PASSPHRASE") or None

    creds = None
    if api_key and api_secret and api_pass:
        creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_pass)

    client = ClobClient(
        host=host,
        chain_id=chain_id,
        key=pk,
        creds=creds,
        signature_type=sig_type,
        funder=funder,
    )

    if creds is None:
        # 一次性：生成或派生 API creds（注意：create 新 key 会使旧 key 失效）
        api_creds = client.create_or_derive_api_creds()
        client.set_api_creds(api_creds)
        # 你需要把它们保存到 .env（不要提交到 git）
        print("Generated/derived API creds (save to .env, keep secret):")
        print(json.dumps(api_creds, ensure_ascii=False, indent=2))

    return client


def main() -> int:
    load_dotenv(".env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--usdc", type=float, default=1.0, help="BUY 预算（USDC），默认 1")
    ap.add_argument("--live", action="store_true", help="真的下单（默认 dry-run）")
    ap.add_argument("--post-only", action="store_true", default=True, help="post-only（默认开）")
    ap.add_argument("--order-type", choices=["GTC", "FOK"], default="GTC", help="GTC 限价或 FOK 市价（带最坏价）")
    ap.add_argument("--slippage", type=float, default=0.02, help="最坏价保护（仅 FOK 使用），默认 0.02")
    ap.add_argument("--cancel-after-sec", type=int, default=10, help="GTC 下单后多少秒撤单（默认 10）")
    args = ap.parse_args()

    m = resolve_eth_2200_yes_market()
    best_bid, best_ask = get_best_bid_ask(m.token_id_yes)

    # 选一个“不会穿价”的价格作为 post-only GTC 的报价：
    # - 如果有 bid/ask：尝试在 best_bid 上方一个 tick 提升挂单优先级，但必须 < best_ask
    # - 如果缺一边：退化为用 /price(side=BUY) 取一个参考价
    if best_bid is not None and best_ask is not None:
        candidate = quantize_up(best_bid + m.tick_size, m.tick_size)
        if candidate >= best_ask:
            candidate = quantize_down(best_ask - m.tick_size, m.tick_size)
        limit_price = max(m.tick_size, candidate)
    else:
        px = _get_json(f"{CLOB}/price", params={"token_id": m.token_id_yes, "side": "BUY"})
        limit_price = quantize_down(float(px["price"]), m.tick_size)

    implied_shares = args.usdc / max(1e-9, limit_price)
    plan = {
        "question": m.question,
        "condition_id": m.condition_id,
        "token_id_yes": m.token_id_yes,
        "tick_size": m.tick_size,
        "neg_risk": m.neg_risk,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "order": {
            "side": "BUY",
            "usdc": args.usdc,
            "limit_price": limit_price,
            "implied_shares": implied_shares,
            "min_order_size_shares": m.min_order_size,
            "post_only": bool(args.post_only),
            "order_type": args.order_type,
        },
    }

    print("PLAN:")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    # 重要：Polymarket 规则里，BUY 的 size 表示“花费的美元金额”，SELL 的 size 表示“份额数量”。
    # minimum_order_size 在 CLOB market 上对 BUY 来说就是最小花费阈值（USDC），所以 1 USDC 会被拒。
    if float(args.usdc) < float(m.min_order_size):
        print(f"ABORT: BUY usdc({float(args.usdc):.6f}) < minimum_order_size({m.min_order_size})")
        print("HINT: 把 --usdc 提高到 >= minimum_order_size（通常是 5）再试。")
        return 2

    if not args.live:
        print("dry-run: not placing any order (pass --live to place).")
        return 0

    client = build_client(host=CLOB, chain_id=CHAIN_ID)

    # tick_size/neg_risk 作为 options（必须）
    options = {"tick_size": str(m.tick_size), "neg_risk": bool(m.neg_risk)}

    if args.order_type == "GTC":
        opts = PartialCreateOrderOptions(tick_size=tick_size_to_literal(m.tick_size), neg_risk=bool(m.neg_risk))
        signed = client.create_order(OrderArgs(token_id=m.token_id_yes, price=limit_price, size=float(args.usdc), side=BUY), opts)
        resp = client.post_order(signed, OrderType.GTC, post_only=bool(args.post_only))
        print("POST_ORDER_RESP:")
        print(json.dumps(resp, ensure_ascii=False, indent=2))

        order_id = resp.get("orderID") or resp.get("orderId") or resp.get("order_id")
        if order_id:
            time.sleep(max(0, int(args.cancel_after_sec)))
            cancel = client.cancel_orders([order_id])
            print("CANCEL_RESP:")
            print(json.dumps(cancel, ensure_ascii=False, indent=2))
        return 0

    # FOK 市价：amount=花费 USDC，price=最坏价保护（例如 best_ask + slippage）
    worst = float(limit_price + float(args.slippage))
    opts = PartialCreateOrderOptions(tick_size=tick_size_to_literal(m.tick_size), neg_risk=bool(m.neg_risk))
    mkt_signed = client.create_market_order(
        MarketOrderArgs(
            token_id=m.token_id_yes,
            amount=float(args.usdc),
            side=BUY,
            price=worst,
            order_type=OrderType.FOK,
        ),
        opts,
    )
    resp = client.post_order(mkt_signed, OrderType.FOK)
    print("POST_MARKET_ORDER_RESP:")
    print(json.dumps(resp, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

