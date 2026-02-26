from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


def quantile(sorted_values: list[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    n = len(sorted_values)
    # linear interpolation between closest ranks
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    if hi == lo:
        return float(sorted_values[lo])
    w = pos - lo
    return float(sorted_values[lo] * (1 - w) + sorted_values[hi] * w)


def summarize(values: Iterable[float]) -> dict[str, Optional[float]]:
    vs = [float(x) for x in values]
    vs.sort()
    if not vs:
        return {"count": 0, "min": None, "p10": None, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}
    s = sum(vs)
    return {
        "count": len(vs),
        "min": float(vs[0]),
        "p10": quantile(vs, 0.10),
        "p50": quantile(vs, 0.50),
        "p90": quantile(vs, 0.90),
        "p99": quantile(vs, 0.99),
        "max": float(vs[-1]),
        "mean": float(s / len(vs)),
    }


@dataclass(frozen=True)
class DrawdownResult:
    mdd_abs: float
    mdd_pct: Optional[float]
    peak: float
    trough: float


def max_drawdown(equity: list[float], baseline: Optional[float] = None) -> DrawdownResult:
    if not equity:
        return DrawdownResult(mdd_abs=0.0, mdd_pct=None, peak=0.0, trough=0.0)
    peak = equity[0]
    peak_i = 0
    mdd = 0.0
    trough_at_mdd = equity[0]
    peak_at_mdd = equity[0]

    for i, x in enumerate(equity):
        if x > peak:
            peak = x
            peak_i = i
        dd = peak - x
        if dd > mdd:
            mdd = dd
            trough_at_mdd = x
            peak_at_mdd = peak

    if baseline is None:
        mdd_pct = None
    else:
        denom = max(1e-9, float(baseline) + float(peak_at_mdd))
        mdd_pct = float(mdd / denom)
    return DrawdownResult(mdd_abs=float(mdd), mdd_pct=mdd_pct, peak=float(peak_at_mdd), trough=float(trough_at_mdd))

