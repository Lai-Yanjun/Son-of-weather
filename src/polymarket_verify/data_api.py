from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import httpx

from .models import ActivityTrade, ClosedPosition, Position


@dataclass(frozen=True)
class DataApiClient:
    base_url: str = "https://data-api.polymarket.com"
    timeout_sec: float = 20.0

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_sec,
            headers={
                "accept": "application/json",
                # 尽量降低 Cloudflare/边缘缓存的不确定性（仍可能被节流）
                "user-agent": "pm-copytrader-verify/0.1",
            },
        )

    def get_activity(
        self,
        *,
        user: str,
        limit: int = 500,
        offset: int = 0,
        types: str = "TRADE",
        sort_by: str = "TIMESTAMP",
        sort_direction: str = "DESC",
        side: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "user": user,
            "limit": limit,
            "offset": offset,
            "type": types,
            "sortBy": sort_by,
            "sortDirection": sort_direction,
        }
        if side:
            params["side"] = side
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        with self._client() as c:
            r = c.get("/activity", params=params)
            r.raise_for_status()
            return r.json()

    def iter_activity_trades(
        self,
        *,
        user: str,
        max_trades: int = 2000,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Iterable[ActivityTrade]:
        fetched = 0
        limit = 500
        offset = 0
        while fetched < max_trades:
            batch_raw = self.get_activity(
                user=user,
                limit=min(limit, max_trades - fetched),
                offset=offset,
                types="TRADE",
                start=start,
                end=end,
                sort_by="TIMESTAMP",
                sort_direction="DESC",
            )
            if not batch_raw:
                break
            for item in batch_raw:
                if item.get("type") != "TRADE":
                    continue
                yield ActivityTrade.model_validate(item)
                fetched += 1
                if fetched >= max_trades:
                    break
            offset += len(batch_raw)
            if len(batch_raw) < limit:
                break

    def get_positions(self, *, user: str) -> list[Position]:
        with self._client() as c:
            r = c.get("/positions", params={"user": user})
            r.raise_for_status()
            data = r.json()
        return [Position.model_validate(x) for x in data]

    def get_value(self, *, user: str) -> float:
        with self._client() as c:
            r = c.get("/value", params={"user": user})
            r.raise_for_status()
            data = r.json()
        if not data:
            return 0.0
        return float(data[0].get("value", 0.0))

    def get_closed_positions_page(self, *, user: str, limit: int, offset: int) -> list[dict[str, Any]]:
        with self._client() as c:
            r = c.get("/closed-positions", params={"user": user, "limit": limit, "offset": offset})
            r.raise_for_status()
            return r.json()

    def iter_closed_positions(self, *, user: str, max_items: int = 3000) -> Iterable[ClosedPosition]:
        fetched = 0
        limit = 500
        offset = 0
        while fetched < max_items:
            batch_raw = self.get_closed_positions_page(
                user=user,
                limit=min(limit, max_items - fetched),
                offset=offset,
            )
            if not batch_raw:
                break
            for item in batch_raw:
                yield ClosedPosition.model_validate(item)
                fetched += 1
                if fetched >= max_items:
                    break
            offset += len(batch_raw)
            if len(batch_raw) < limit:
                break

