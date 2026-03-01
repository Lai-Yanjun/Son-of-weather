from __future__ import annotations

"""
下单连通性自检（树莓派用）：
- 默认 dry-run（不下单）
- 加 --live 才会真实下单，并在数秒后撤单

示例：
  ./.venv/bin/python check_order_connectivity.py
  ./.venv/bin/python check_order_connectivity.py --live --usdc 2 --cancel-after-sec 5
"""

import argparse
import json
import os
import time
import urllib.request
from typing import Any

import httpx
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType, PartialCreateOrderOptions
from py_clob_client.exceptions import PolyApiException
from py_clob_client.order_builder.constants import BUY


DATA_API = "https://data-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def tick_size_to_literal(tick: float) -> str:
    mapping = {0.1: "0.1", 0.01: "0.01", 0.001: "0.001", 0.0001: "0.0001"}
    for k, v in mapping.items():
        if abs(float(tick) - k) <= 1e-9:
            return v
    return f"{tick:.4f}".rstrip("0").rstrip(".")


def _get_json(url: str, *, params: dict[str, Any] | None = None) -> Any:
    r = httpx.get(url, params=params, timeout=20.0, headers={"accept": "application/json", "user-agent": "pm-order-check/0.1"})
    r.raise_for_status()
    return r.json()


def _setup_proxy_env(*, proxy: str | None, http_proxy: str | None, https_proxy: str | None, no_proxy: str) -> None:
    if proxy:
        os.environ["http_proxy"] = str(proxy)
        os.environ["https_proxy"] = str(proxy)
        os.environ["HTTP_PROXY"] = str(proxy)
        os.environ["HTTPS_PROXY"] = str(proxy)
    if http_proxy:
        os.environ["http_proxy"] = str(http_proxy)
        os.environ["HTTP_PROXY"] = str(http_proxy)
    if https_proxy:
        os.environ["https_proxy"] = str(https_proxy)
        os.environ["HTTPS_PROXY"] = str(https_proxy)
    os.environ["no_proxy"] = str(no_proxy)
    os.environ["NO_PROXY"] = str(no_proxy)


def _probe_public_ip() -> str:
    try:
        with urllib.request.urlopen("https://ifconfig.me/ip", timeout=10) as r:
            return r.read().decode("utf-8", errors="ignore").strip()
    except Exception as e:
        return f"(ip_probe_failed:{type(e).__name__})"


def build_client() -> ClobClient:
    pk = os.getenv("PRIVATE_KEY")
    if not pk:
        raise RuntimeError("missing PRIVATE_KEY in .env")
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
        chain_id=137,
        key=pk,
        creds=creds,
        signature_type=sig_type,
        funder=funder,
    )
    if creds is None:
        client.set_api_creds(client.create_or_derive_api_creds())
    return client


def pick_trade_token(*, user: str, scan: int) -> tuple[str, float, float, bool]:
    raw = _get_json(
        f"{DATA_API}/activity",
        params={"user": user, "limit": int(scan), "offset": 0, "type": "TRADE", "sortBy": "TIMESTAMP", "sortDirection": "DESC"},
    )
    for x in raw or []:
        if x.get("type") != "TRADE" or str(x.get("side") or "").upper() != "BUY":
            continue
        condition_id = str(x["conditionId"])
        outcome_idx = int(x["outcomeIndex"])
        m = _get_json(f"{CLOB}/markets/{condition_id}")
        if bool(m.get("closed")) or (m.get("accepting_orders") is False):
            continue
        tokens = m.get("tokens") or []
        if outcome_idx < 0 or outcome_idx >= len(tokens):
            continue
        token_id = str(tokens[outcome_idx].get("token_id") or tokens[outcome_idx].get("tokenId") or "")
        if not token_id:
            continue
        ask = _get_json(f"{CLOB}/price", params={"token_id": token_id, "side": "BUY"})
        best_ask = float(ask["price"])
        tick = float(m.get("minimum_tick_size") or m.get("tick_size") or 0.001)
        neg_risk = bool(m.get("neg_risk") or m.get("negative_risk") or m.get("negativeRisk") or False)
        return token_id, best_ask, tick, neg_risk
    raise RuntimeError("no tradable BUY found in recent activities")


def main() -> int:
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="0x6297b93ea37ff92a57fd636410f3b71ebf74517e")
    ap.add_argument("--scan", type=int, default=100)
    ap.add_argument("--usdc", type=float, default=2.0)
    ap.add_argument("--cancel-after-sec", type=int, default=5)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--proxy", default=None, help="同时设置 HTTP/HTTPS 代理，例如 http://127.0.0.1:7890")
    ap.add_argument("--http-proxy", default=None, help="仅设置 HTTP 代理")
    ap.add_argument("--https-proxy", default=None, help="仅设置 HTTPS 代理")
    ap.add_argument("--no-proxy", default="localhost,127.0.0.1", help="NO_PROXY，默认 localhost,127.0.0.1")
    args = ap.parse_args()

    _setup_proxy_env(
        proxy=args.proxy,
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy,
        no_proxy=str(args.no_proxy),
    )
    print("NETWORK:")
    print(
        json.dumps(
            {
                "http_proxy": os.getenv("http_proxy") or os.getenv("HTTP_PROXY"),
                "https_proxy": os.getenv("https_proxy") or os.getenv("HTTPS_PROXY"),
                "no_proxy": os.getenv("no_proxy") or os.getenv("NO_PROXY"),
                "egress_ip": _probe_public_ip(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    token_id, best_ask, tick, neg_risk = pick_trade_token(user=str(args.user).lower(), scan=int(args.scan))
    maker_price = max(float(tick), float(best_ask) - float(tick))
    shares = float(args.usdc) / max(1e-12, float(maker_price))
    plan = {
        "token_id": token_id,
        "best_ask": best_ask,
        "tick_size": tick,
        "maker_price": maker_price,
        "usdc": float(args.usdc),
        "shares": shares,
        "neg_risk": neg_risk,
    }
    print("PLAN:")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if not args.live:
        print("RESULT: dry-run only (add --live to place/cancel)")
        return 0

    client = build_client()
    opts = PartialCreateOrderOptions(tick_size=tick_size_to_literal(float(tick)), neg_risk=bool(neg_risk))
    try:
        signed = client.create_order(
            OrderArgs(token_id=str(token_id), price=float(maker_price), size=float(shares), side=BUY),
            opts,
        )
        resp = client.post_order(signed, OrderType.GTC, post_only=True)
        print("POST_ORDER_RESP:")
        print(json.dumps(resp, ensure_ascii=False, indent=2))
        order_id = resp.get("orderID") or resp.get("orderId") or resp.get("order_id")
        if not order_id:
            print("RESULT: failed (no order_id)")
            return 2
        time.sleep(max(0, int(args.cancel_after_sec)))
        cancel = client.cancel_orders([str(order_id)])
        print("CANCEL_RESP:")
        print(json.dumps(cancel, ensure_ascii=False, indent=2))
        print("RESULT: success (api path works)")
        return 0
    except PolyApiException as e:
        status = getattr(e, "status_code", None)
        if status == 403:
            print("RESULT: failed (GEO_BLOCK_403)")
        else:
            print(f"RESULT: failed (POLY_API_{status or 'ERROR'})")
        print(str(getattr(e, "error_message", None) or repr(e)))
        return 3
    except Exception as e:
        print("RESULT: failed (RUNTIME_ERROR)")
        print(repr(e))
        return 4


if __name__ == "__main__":
    raise SystemExit(main())

