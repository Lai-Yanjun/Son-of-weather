from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, MarketOrderArgs, OrderArgs, OrderType, PartialCreateOrderOptions
from py_clob_client.exceptions import PolyApiException
from py_clob_client.order_builder.constants import BUY, SELL

from .clob_public import ClobPublicClient
from .data_api import DataApiClient
from .state_db import StateDB


def _now_ms() -> int:
    return int(time.time() * 1000)


def tick_size_to_literal(tick: float) -> str:
    mapping = {0.1: "0.1", 0.01: "0.01", 0.001: "0.001", 0.0001: "0.0001"}
    for k, v in mapping.items():
        if abs(float(tick) - k) <= 1e-9:
            return v
    return f"{tick:.4f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class Action:
    run_id: int
    seen_ts: int
    trade_ts: int
    condition_id: str
    outcome_index: int
    token_id: str
    side: str  # BUY/SELL
    # sizing
    usdc: float  # for BUY: spend; for SELL: reference budget (used to compute shares)
    shares: float  # for SELL: shares to sell; for BUY: implied shares
    # pricing
    price: float  # limit price (GTC) or reference for FOK worst
    tick_size: float
    neg_risk: bool


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    action_side: str
    order_id: Optional[str]
    reason: Optional[str]
    detail: Optional[str]
    ack_ts: int
    seen_to_ack_ms: int
    cancel_resp: Optional[dict[str, Any]]


class ShadowExecutor:
    def __init__(self, *, db: StateDB) -> None:
        self._db = db

    def execute(self, action: Action) -> ExecResult:
        ack_ts = int(time.time())
        try:
            if action.side.upper() == "BUY":
                fill = self._db.apply_shadow_fill(
                    token_id=action.token_id,
                    condition_id=action.condition_id,
                    outcome_index=action.outcome_index,
                    side="BUY",
                    shares=float(action.shares),
                    price=float(action.price),
                )
                _ = fill
                return ExecResult(
                    ok=True,
                    action_side="BUY",
                    order_id=None,
                    reason=None,
                    detail=None,
                    ack_ts=ack_ts,
                    seen_to_ack_ms=int((ack_ts - int(action.seen_ts)) * 1000),
                    cancel_resp=None,
                )

            if action.side.upper() == "SELL":
                fill = self._db.apply_shadow_fill(
                    token_id=action.token_id,
                    condition_id=action.condition_id,
                    outcome_index=action.outcome_index,
                    side="SELL",
                    shares=float(action.shares),
                    price=float(action.price),
                )
                _ = fill
                return ExecResult(
                    ok=True,
                    action_side="SELL",
                    order_id=None,
                    reason=None,
                    detail=None,
                    ack_ts=ack_ts,
                    seen_to_ack_ms=int((ack_ts - int(action.seen_ts)) * 1000),
                    cancel_resp=None,
                )

            return ExecResult(
                ok=False,
                action_side=str(action.side),
                order_id=None,
                reason="UNKNOWN_SIDE",
                detail=str(action.side),
                ack_ts=ack_ts,
                seen_to_ack_ms=int((ack_ts - int(action.seen_ts)) * 1000),
                cancel_resp=None,
            )
        except Exception as e:
            return ExecResult(
                ok=False,
                action_side=str(action.side),
                order_id=None,
                reason="SHADOW_EXEC_ERROR",
                detail=repr(e),
                ack_ts=ack_ts,
                seen_to_ack_ms=int((ack_ts - int(action.seen_ts)) * 1000),
                cancel_resp=None,
            )


class LiveExecutor:
    def __init__(
        self,
        *,
        clob_base: str = "https://clob.polymarket.com",
        data_api_base: str = "https://data-api.polymarket.com",
        chain_id: int = 137,
        timeout_sec: float = 20.0,
        post_only: bool = True,
        taker: bool = False,
        order_type: str = "GTC",
        cancel_after_sec: int = 10,
        slippage: float = 0.02,
    ) -> None:
        load_dotenv(".env")
        self._post_only = bool(post_only) and (not bool(taker))
        self._taker = bool(taker)
        self._order_type = str(order_type).upper()
        self._cancel_after_sec = int(cancel_after_sec)
        self._slippage = float(slippage)

        pk = os.getenv("PRIVATE_KEY")
        if not pk:
            raise RuntimeError("missing PRIVATE_KEY in .env")
        sig_type = int(os.getenv("SIGNATURE_TYPE", "0"))
        funder = os.getenv("FUNDER_ADDRESS") or None
        self._funder = funder

        api_key = os.getenv("POLY_API_KEY") or None
        api_secret = os.getenv("POLY_SECRET") or None
        api_pass = os.getenv("POLY_PASSPHRASE") or None
        if not (api_key and api_secret and api_pass):
            raise RuntimeError("missing POLY_API_KEY/POLY_SECRET/POLY_PASSPHRASE in .env (run gen_clob_api_creds.py)")

        creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_pass)
        self._client = ClobClient(host=clob_base, chain_id=chain_id, key=pk, creds=creds, signature_type=sig_type, funder=funder)

        self._clob_pub = ClobPublicClient(base_url=clob_base, timeout_sec=timeout_sec)
        self._data = DataApiClient(base_url=data_api_base, timeout_sec=timeout_sec)

    def close(self) -> None:
        try:
            self._clob_pub.close()
        finally:
            self._data.close()

    def _available_shares(self, *, token_id: str) -> float:
        if not self._funder:
            return 0.0
        # Data API positions 是公开的；用 funder 地址查询自己账户当前持仓（近似可卖份额）
        try:
            pos = self._data.get_positions(user=str(self._funder).lower())
        except Exception:
            return 0.0
        for p in pos:
            if str(p.asset) == str(token_id):
                return float(p.size)
        return 0.0

    def execute(self, action: Action) -> ExecResult:
        seen_ms = int(action.seen_ts) * 1000
        ack_ts = int(time.time())
        ack_ms = _now_ms()
        cancel_resp: Optional[dict[str, Any]] = None
        try:
            side = action.side.upper()
            opts = PartialCreateOrderOptions(tick_size=tick_size_to_literal(action.tick_size), neg_risk=bool(action.neg_risk))

            if side == "BUY":
                if self._order_type == "FOK":
                    worst = float(action.price + self._slippage)
                    signed = self._client.create_market_order(
                        MarketOrderArgs(
                            token_id=action.token_id,
                            amount=float(action.usdc),
                            side=BUY,
                            price=float(worst),
                            order_type=OrderType.FOK,
                        ),
                        opts,
                    )
                    resp = self._client.post_order(signed, OrderType.FOK)
                else:
                    signed = self._client.create_order(
                        OrderArgs(token_id=action.token_id, price=float(action.price), size=float(action.usdc), side=BUY),
                        opts,
                    )
                    resp = self._client.post_order(signed, OrderType.GTC, post_only=bool(self._post_only))

                ack_ts = int(time.time())
                ack_ms = _now_ms()
                order_id = resp.get("orderID") or resp.get("orderId") or resp.get("order_id")
                if order_id and self._order_type != "FOK":
                    time.sleep(max(0, int(self._cancel_after_sec)))
                    cancel_resp = self._client.cancel_orders([str(order_id)])
                return ExecResult(
                    ok=bool(resp.get("success", True)),
                    action_side="BUY",
                    order_id=str(order_id) if order_id else None,
                    reason=None if resp.get("success", True) else "POST_ORDER_FAILED",
                    detail=None if resp.get("success", True) else json.dumps(resp, ensure_ascii=False),
                    ack_ts=ack_ts,
                    seen_to_ack_ms=int(max(0, ack_ms - seen_ms)),
                    cancel_resp=cancel_resp,
                )

            if side == "SELL":
                avail = self._available_shares(token_id=action.token_id)
                if avail <= 0:
                    return ExecResult(
                        ok=False,
                        action_side="SELL",
                        order_id=None,
                        reason="NO_POSITION_TO_SELL",
                        detail="available_shares=0",
                        ack_ts=ack_ts,
                        seen_to_ack_ms=int(_now_ms() - t0),
                        cancel_resp=None,
                    )
                sell_shares = min(float(action.shares), float(avail))
                signed = self._client.create_order(
                    OrderArgs(token_id=action.token_id, price=float(action.price), size=float(sell_shares), side=SELL),
                    opts,
                )
                resp = self._client.post_order(signed, OrderType.GTC, post_only=bool(self._post_only))
                ack_ts = int(time.time())
                ack_ms = _now_ms()
                order_id = resp.get("orderID") or resp.get("orderId") or resp.get("order_id")
                if order_id:
                    time.sleep(max(0, int(self._cancel_after_sec)))
                    cancel_resp = self._client.cancel_orders([str(order_id)])
                return ExecResult(
                    ok=bool(resp.get("success", True)),
                    action_side="SELL",
                    order_id=str(order_id) if order_id else None,
                    reason=None if resp.get("success", True) else "POST_ORDER_FAILED",
                    detail=None if resp.get("success", True) else json.dumps(resp, ensure_ascii=False),
                    ack_ts=ack_ts,
                    seen_to_ack_ms=int(max(0, ack_ms - seen_ms)),
                    cancel_resp=cancel_resp,
                )

            return ExecResult(
                ok=False,
                action_side=str(action.side),
                order_id=None,
                reason="UNKNOWN_SIDE",
                detail=str(action.side),
                ack_ts=ack_ts,
                seen_to_ack_ms=int(max(0, _now_ms() - seen_ms)),
                cancel_resp=None,
            )
        except PolyApiException as e:
            ack_ts = int(time.time())
            status = getattr(e, "status_code", None)
            reason = f"POLY_API_{status}" if status else "POLY_API_ERROR"
            if status == 403:
                reason = "GEO_BLOCK_403"
            return ExecResult(
                ok=False,
                action_side=str(action.side).upper(),
                order_id=None,
                reason=reason,
                detail=str(getattr(e, "error_message", None) or repr(e)),
                ack_ts=ack_ts,
                seen_to_ack_ms=int(max(0, _now_ms() - seen_ms)),
                cancel_resp=None,
            )
        except Exception as e:
            ack_ts = int(time.time())
            return ExecResult(
                ok=False,
                action_side=str(action.side).upper(),
                order_id=None,
                reason="LIVE_EXEC_ERROR",
                detail=repr(e),
                ack_ts=ack_ts,
                seen_to_ack_ms=int(max(0, _now_ms() - seen_ms)),
                cancel_resp=None,
            )

