from __future__ import annotations

from typing import Any


def make_parameter_playbook(*, stats: dict[str, Any]) -> list[str]:
    """
    输出“可操作”的参数建议（阶段0：只给建议，不涉及下单实现）。
    """
    ep = stats.get("execution_profile", {}) or {}
    mk = stats.get("mdd_and_k", {}) or {}

    usdc = (ep.get("usdc_size") or {}) if ep else {}
    p50 = usdc.get("p50")
    p90 = usdc.get("p90")
    mx = usdc.get("max")

    bullets: list[str] = []

    bullets.append("**白名单**：初期只允许你“理解且愿意承担风险”的市场（建议先白名单 `eventSlug` 为天气相关的那一类），其它一律只记录不跟。")
    bullets.append("**合并窗口**：`merge_window_sec=2` 起步（你样本里存在同秒连打），避免把一次建仓拆成多笔导致滑点与成交不确定性上升。")
    bullets.append("**滑点阈值**：`slippage_max=0.02`（2¢）起步；若你发现经常被跳过且不影响结果，可放宽到 0.03；若被追价，收紧到 0.01。")

    if p50 is not None and p90 is not None and mx is not None:
        bullets.append(
            f"**单笔金额门槛**：他的单笔 usdcSize 分布 p50≈{p50:.4f}、p90≈{p90:.4f}、max≈{mx:.4f}。"
            "你不必跟随极小额“试单/噪声”，建议把 `MinPerTrade` 设在你能接受的最小成本（例如 1–5 USDC），低于则忽略。"
        )
    else:
        bullets.append("**单笔金额门槛**：建议 `MinPerTrade` 设在 1–5 USDC（过滤试单/噪声），`MaxPerTrade` 先设 10–30 USDC（小额实盘时）。")

    bullets.append("**敞口复制（核心）**：按“成本敞口/风险敞口”复制，不按 shares：E_his = shares * avgPrice；E_you = min(k * E_his, MaxPerTrade)；再用当前/限价价格反推 shares_you = E_you / p_now。")
    bullets.append("**单市场/总敞口**：建议 `max_market_exposure` ≤ `max_total_cost_exposure` 的 10–25%（防止集中到某一个城市/某一天的天气）。")

    mdd_proxy = mk.get("mdd_proxy_abs")
    if isinstance(mdd_proxy, (int, float)) and mdd_proxy > 0:
        bullets.append(
            f"**熔断（单日亏损）**：你可以把 `daily_loss_limit` 设在 {mdd_proxy:.2f} USDC 量级或更小（取决于你愿意承受的回撤）。"
            "触发后：只记录不下单，直到人工解除。"
        )
    else:
        bullets.append("**熔断（单日亏损）**：建议先设置一个明确的 USDC 额度（例如 10–50 USDC），触发后只记录不下单。")

    bullets.append("**轮询**：先用 3s 左右（带抖动）+ 指数退避；当你切实盘后再根据 429/延迟与滑点跳过率去微调。")
    bullets.append("**逐笔复制 vs 净持仓同步**：若你发现他经常 10 分钟内来回手，逐笔复制会产生不必要的摩擦成本；此时应优先考虑“净持仓同步”模式（更稳）。")

    return bullets

