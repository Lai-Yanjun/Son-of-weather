from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx


DEFAULT_CLOB_BASE = "https://clob.polymarket.com"


@dataclass(frozen=True)
class MarketToken:
    token_id: str
    outcome: Optional[str]
    outcome_index: int


@dataclass(frozen=True)
class MarketInfo:
    condition_id: str
    question: str
    minimum_order_size: float
    tick_size: float
    neg_risk: bool
    tokens: list[MarketToken]


class ClobPublicClient:
    def __init__(self, *, base_url: str = DEFAULT_CLOB_BASE, timeout_sec: float = 20.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_sec

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"accept": "application/json", "user-agent": "pm-shadow/0.1"},
        )

    def get_market_info(self, condition_id: str) -> MarketInfo:
        with self._client() as c:
            r = c.get(f"/markets/{condition_id}")
            r.raise_for_status()
            m: dict[str, Any] = r.json()

        raw_tokens = m.get("tokens") or []
        tokens: list[MarketToken] = []
        for i, t in enumerate(raw_tokens):
            if not isinstance(t, dict):
                continue
            token_id = str(t.get("token_id") or t.get("tokenId") or "")
            if not token_id:
                continue
            outcome = t.get("outcome")
            if outcome is not None:
                outcome = str(outcome)

            # 多数市场 tokens 顺序与 outcomeIndex 对齐；如果未来 API 给出 outcome_index 字段，也兼容。
            outcome_index = t.get("outcome_index")
            if outcome_index is None:
                outcome_index = t.get("outcomeIndex")
            idx = int(outcome_index) if outcome_index is not None else int(i)

            tokens.append(MarketToken(token_id=token_id, outcome=outcome, outcome_index=idx))

        tokens.sort(key=lambda x: x.outcome_index)

        q = m.get("question") or m.get("title") or ""
        minimum_order_size = float(m.get("minimum_order_size") or m.get("minimumOrderSize") or 0.0)
        tick_size = float(m.get("tick_size") or m.get("tickSize") or 0.001)
        neg_risk = bool(m.get("neg_risk") or m.get("negative_risk") or m.get("negativeRisk") or False)

        return MarketInfo(
            condition_id=str(condition_id),
            question=str(q),
            minimum_order_size=minimum_order_size,
            tick_size=tick_size,
            neg_risk=neg_risk,
            tokens=tokens,
        )

    def resolve_token_id(self, *, condition_id: str, outcome_index: int, outcome: str | None = None) -> str:
        mi = self.get_market_info(condition_id)
        for t in mi.tokens:
            if t.outcome_index == int(outcome_index):
                return t.token_id
        if outcome:
            outcome_l = str(outcome).strip().lower()
            for t in mi.tokens:
                if (t.outcome or "").strip().lower() == outcome_l:
                    return t.token_id
        raise KeyError(f"无法从 CLOB market 解析 token_id: condition_id={condition_id} outcome_index={outcome_index}")

    def get_best_prices(self, token_id: str) -> tuple[float, float]:
        # /price: side=BUY 给出“买入该 token 的最佳价”（通常可视为 best ask）
        #        side=SELL 给出“卖出该 token 的最佳价”（通常可视为 best bid）
        with self._client() as c:
            buy = c.get("/price", params={"token_id": token_id, "side": "BUY"})
            buy.raise_for_status()
            sell = c.get("/price", params={"token_id": token_id, "side": "SELL"})
            sell.raise_for_status()
        bj = buy.json()
        sj = sell.json()
        return float(bj["price"]), float(sj["price"])

