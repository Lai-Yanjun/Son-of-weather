from __future__ import annotations

from typing import Any, Iterable, Optional

import httpx

from .models import ActivityTrade, ClosedPosition, Position


class DataApiClient:
    def __init__(self, *, base_url: str = "https://data-api.polymarket.com", timeout_sec: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_sec,
            headers={
                "accept": "application/json",
                # 尽量降低 Cloudflare/边缘缓存的不确定性（仍可能被节流）
                "user-agent": "pm-copytrader-verify/0.1",
            },
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "DataApiClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

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

        r = self._http.get("/activity", params=params)
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
        r = self._http.get("/positions", params={"user": user})
        r.raise_for_status()
        data = r.json()
        return [Position.model_validate(x) for x in data]

    def get_value(self, *, user: str) -> float:
        r = self._http.get("/value", params={"user": user})
        r.raise_for_status()
        data = r.json()
        if not data:
            return 0.0
        return float(data[0].get("value", 0.0))

    def get_closed_positions_page(self, *, user: str, limit: int, offset: int) -> list[dict[str, Any]]:
        r = self._http.get("/closed-positions", params={"user": user, "limit": limit, "offset": offset})
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

