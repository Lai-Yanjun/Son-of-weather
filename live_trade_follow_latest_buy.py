from __future__ import annotations

"""
实盘测试（默认 dry-run）：跟随某地址最近一笔 BUY 成交，尝试对同一 token 下一个小额 BUY。

重要：
- 默认 dry-run，不会下单；只有 --live 才会真的发单。
- 该脚本会跳过已关闭/不接单/无 orderbook 的市场，并从最近成交里继续往前找。

Windows:
  .\\.venv\\Scripts\\python live_trade_follow_latest_buy.py --usdc 5
  .\\.venv\\Scripts\\python live_trade_follow_latest_buy.py --usdc 5 --live
Linux:
  ./.venv/bin/python live_trade_follow_latest_buy.py --usdc 5
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
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType, PartialCreateOrderOptions
from py_clob_client.order_builder.constants import BUY


DATA_API = "https://data-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
CHAIN_ID = 137


@dataclass(frozen=True)
class PickedTrade:
    condition_id: str
    outcome_index: int
    price: float
    usdc_size: float
    timestamp: int
    transaction_hash: str
    title: Optional[str]


@dataclass(frozen=True)
class PickedMarket:
    condition_id: str
    token_id: str
    question: str
    minimum_order_size: float
    tick_size: float
    neg_risk: bool
    best_ask: float
    best_bid: float
    maker_buy_price: float


def _get_json(url: str, *, params: dict[str, Any] | None = None) -> Any:
    # 轻量重试：应对偶发的 WinError 10053/连接中断/临时 5xx
    last_err: Exception | None = None
    for i in range(3):
        try:
            r = httpx.get(
                url,
                params=params,
                timeout=20.0,
                headers={"accept": "application/json", "user-agent": "pm-follow-latest/0.1"},
            )
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            last_err = e
            time.sleep(0.25 * (2**i))
    raise last_err or RuntimeError("request failed")


def tick_size_to_literal(tick: float) -> str:
    mapping = {0.1: "0.1", 0.01: "0.01", 0.001: "0.001", 0.0001: "0.0001"}
    for k, v in mapping.items():
        if abs(float(tick) - k) <= 1e-9:
            return v
    return f"{tick:.4f}".rstrip("0").rstrip(".")


def build_client() -> ClobClient:
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
        host=CLOB,
        chain_id=CHAIN_ID,
        key=pk,
        creds=creds,
        signature_type=sig_type,
        funder=funder,
    )

    if creds is None:
        api_creds = client.create_or_derive_api_creds()
        client.set_api_creds(api_creds)
        print("Generated/derived API creds (save to .env, keep secret):")
        print(json.dumps(api_creds, ensure_ascii=False, indent=2))

    return client


def pick_latest_buy(*, user: str, limit: int) -> list[PickedTrade]:
    raw = _get_json(
        f"{DATA_API}/activity",
        params={"user": user, "limit": int(limit), "offset": 0, "type": "TRADE", "sortBy": "TIMESTAMP", "sortDirection": "DESC"},
    )
    out: list[PickedTrade] = []
    for x in raw or []:
        if x.get("type") != "TRADE":
            continue
        if str(x.get("side") or "").upper() != "BUY":
            continue
        out.append(
            PickedTrade(
                condition_id=str(x["conditionId"]),
                outcome_index=int(x["outcomeIndex"]),
                price=float(x["price"]),
                usdc_size=float(x["usdcSize"]),
                timestamp=int(x["timestamp"]),
                transaction_hash=str(x["transactionHash"]),
                title=x.get("title"),
            )
        )
    return out


def resolve_trade_to_market(t: PickedTrade) -> PickedMarket:
    m = _get_json(f"{CLOB}/markets/{t.condition_id}")
    if bool(m.get("closed")) or (m.get("accepting_orders") is False):
        raise RuntimeError("MARKET_CLOSED_OR_NOT_ACCEPTING")
    if not bool(m.get("enable_order_book", True)):
        raise RuntimeError("ORDERBOOK_DISABLED")

    tokens = m.get("tokens") or []
    if not tokens or t.outcome_index < 0 or t.outcome_index >= len(tokens):
        raise RuntimeError("TOKEN_INDEX_OUT_OF_RANGE")
    token_id = str(tokens[int(t.outcome_index)].get("token_id") or tokens[int(t.outcome_index)].get("tokenId"))
    if not token_id:
        raise RuntimeError("TOKEN_ID_MISSING")

    # quote
    try:
        buy = _get_json(f"{CLOB}/price", params={"token_id": token_id, "side": "BUY"})
        sell = _get_json(f"{CLOB}/price", params={"token_id": token_id, "side": "SELL"})
        # docs: BUY -> best ask, SELL -> best bid
        best_ask = float(buy["price"])
        best_bid = float(sell["price"])
    except httpx.HTTPStatusError as e:
        # /price 对无 orderbook 的 token 可能返回 404（带 error msg）
        raise RuntimeError(f"QUOTE_FAILED:{e.response.status_code}") from e

    tick = float(m.get("minimum_tick_size") or m.get("tick_size") or 0.001)
    maker_buy_price = max(tick, float(best_ask) - tick)

    return PickedMarket(
        condition_id=str(t.condition_id),
        token_id=token_id,
        question=str(m.get("question") or m.get("title") or ""),
        minimum_order_size=float(m.get("minimum_order_size") or 0.0),
        tick_size=tick,
        neg_risk=bool(m.get("neg_risk") or m.get("negative_risk") or m.get("negativeRisk") or False),
        best_ask=best_ask,
        best_bid=best_bid,
        maker_buy_price=maker_buy_price,
    )


def main() -> int:
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="0x6297b93ea37ff92a57fd636410f3b71ebf74517e", help="跟随对象地址（默认 neobrother）")
    ap.add_argument("--scan", type=int, default=80, help="最多向前扫描多少条成交（默认 80）")
    ap.add_argument("--usdc", type=float, default=5.0, help="测试 BUY 花费 USDC（默认 5）")
    ap.add_argument("--live", action="store_true", help="真的下单（默认 dry-run）")
    ap.add_argument("--taker", action="store_true", help="用 taker 方式测试（更可能成交，风险更高）")
    ap.add_argument("--cancel-after-sec", type=int, default=10, help="下单后多少秒撤单（默认 10）")
    args = ap.parse_args()

    trades = pick_latest_buy(user=str(args.user).lower(), limit=int(args.scan))
    if not trades:
        print("ABORT: 最近扫描范围内没有 BUY 成交。")
        return 2

    picked_trade = None
    picked_market = None
    skipped: dict[str, int] = {}
    for t in trades:
        try:
            m = resolve_trade_to_market(t)
            picked_trade = t
            picked_market = m
            break
        except Exception as e:
            k = str(e)
            skipped[k] = skipped.get(k, 0) + 1
            continue

    if not picked_trade or not picked_market:
        print("ABORT: 未找到可交易的最近 BUY（全部被跳过）。")
        print(json.dumps({"skipped": skipped}, ensure_ascii=False, indent=2))
        return 3

    if float(args.usdc) < float(picked_market.minimum_order_size):
        print(f"ABORT: usdc({float(args.usdc):.6f}) < minimum_order_size({picked_market.minimum_order_size})")
        return 4

    post_only = not bool(args.taker)
    price = float(picked_market.best_ask) if bool(args.taker) else float(picked_market.maker_buy_price)

    plan = {
        "picked_trade": picked_trade.__dict__,
        "picked_market": picked_market.__dict__,
        "you": {"side": "BUY", "usdc": float(args.usdc), "price": price, "post_only": post_only, "mode": ("TAKER" if args.taker else "POST_ONLY_MAKER")},
        "skipped": skipped,
    }
    print("PLAN:")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if not args.live:
        print("dry-run: not placing any order (pass --live to place).")
        return 0

    client = build_client()
    opts = PartialCreateOrderOptions(tick_size=tick_size_to_literal(picked_market.tick_size), neg_risk=bool(picked_market.neg_risk))

    signed = client.create_order(
        OrderArgs(token_id=picked_market.token_id, price=float(price), size=float(args.usdc), side=BUY),
        opts,
    )
    resp = client.post_order(signed, OrderType.GTC, post_only=bool(post_only))
    print("POST_ORDER_RESP:")
    print(json.dumps(resp, ensure_ascii=False, indent=2))

    order_id = resp.get("orderID") or resp.get("orderId") or resp.get("order_id")
    if order_id:
        time.sleep(max(0, int(args.cancel_after_sec)))
        cancel = client.cancel_orders([order_id])
        print("CANCEL_RESP:")
        print(json.dumps(cancel, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

