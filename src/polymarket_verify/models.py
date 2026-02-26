from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


Side = Literal["BUY", "SELL"]


class ActivityTrade(BaseModel):
    proxyWallet: str
    timestamp: int
    conditionId: str
    type: Literal["TRADE"]
    size: float
    usdcSize: float
    transactionHash: str
    price: float
    asset: str
    side: Side
    outcomeIndex: int
    title: Optional[str] = None
    slug: Optional[str] = None
    icon: Optional[str] = None
    eventSlug: Optional[str] = None
    outcome: Optional[str] = None


class Position(BaseModel):
    proxyWallet: str
    asset: str
    conditionId: str
    size: float
    avgPrice: float
    initialValue: float
    currentValue: float
    cashPnl: float
    percentPnl: float
    totalBought: float
    realizedPnl: float
    curPrice: float
    title: Optional[str] = None
    slug: Optional[str] = None
    icon: Optional[str] = None
    eventId: Optional[str] = None
    eventSlug: Optional[str] = None
    outcome: Optional[str] = None
    outcomeIndex: int
    oppositeOutcome: Optional[str] = None
    oppositeAsset: Optional[str] = None
    endDate: Optional[str] = None
    negativeRisk: Optional[bool] = None


class ClosedPosition(BaseModel):
    proxyWallet: str
    asset: str
    conditionId: str
    avgPrice: float
    totalBought: float
    realizedPnl: float
    curPrice: float
    title: Optional[str] = None
    slug: Optional[str] = None
    icon: Optional[str] = None
    eventSlug: Optional[str] = None
    outcome: Optional[str] = None
    outcomeIndex: int
    oppositeOutcome: Optional[str] = None
    oppositeAsset: Optional[str] = None
    endDate: Optional[str] = None
    timestamp: int = Field(..., description="closure timestamp (unix seconds)")


class Config(BaseModel):
    target_username: str = Field(..., alias="username")
    target_address: str = Field(..., alias="address")


class RawConfig(BaseModel):
    target: dict[str, Any]
    api: dict[str, Any]
    analysis: dict[str, Any]
    risk_budget_example: dict[str, Any] = Field(default_factory=dict)
    polling_suggestion: dict[str, Any] = Field(default_factory=dict)

