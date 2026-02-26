from __future__ import annotations

from typing import Any


def polling_and_realtime_advice(execution_profile: dict[str, Any], *, suggested_poll_interval_sec: float, jitter_sec: float) -> dict[str, Any]:
    # 这里不做“绝对数值”的承诺：Data API 是近实时索引，延迟随网络/索引负载变化。
    p50 = (execution_profile.get("deltas_sec") or {}).get("p50")
    p90 = (execution_profile.get("deltas_sec") or {}).get("p90")
    min_delta = (execution_profile.get("deltas_sec") or {}).get("min")
    clusters = (execution_profile.get("clusters") or {}).get("p_cluster_ge_2")

    conclusion = (
        "- **结论**：neobrother 的成交频率属于中低频（分钟级间隔为主，但会有同秒连打）。"
        "因此跟单系统 **不需要“毫秒级/亚秒级”超低延迟**，但需要做好 **去重 + 合并窗口 + 滑点保护**，"
        "以及面对 Data API 的 **近实时延迟/乱序/重复** 的鲁棒性。"
    )

    bullets = [
        f"**轮询间隔建议**：先用 `poll_interval={suggested_poll_interval_sec}s`（带 ±{jitter_sec}s 抖动），观察 429/延迟后再微调；通常 2–5s 是比较稳的工程取值。",
        "**Data API 不是严格实时**：activity 来自“上链→索引→可查询”，可能比链上时间晚几秒到几十秒；且同一笔可能重复/乱序返回，所以必须用 `transactionHash+asset+side+usdcSize+timestamp` 去重。",
        "**同秒连打要合并**：既然 `min_delta` 可能为 0，paper/live 都要对同 token 同方向在 1–2s 内的连续成交做合并，否则你会拆成多笔小单、滑点和手续费都会更差。",
        "**滑点阈值一定要开**：近实时轮询意味着你经常会比他更晚看到成交，价格可能已经走了；建议先从 0.01–0.02（1–2¢）起步，宁可跳过也别追高/追低。",
        "**请求频率上限很宽，但别滥用**：官方 Data API general 限额是 1000 req/10s（Cloudflare 节流为排队/延迟）。我们只需要 1–2 个 endpoint，留足裕量即可。",
    ]

    if p50 is not None and p90 is not None:
        bullets.insert(
            0,
            f"**成交间隔参考**：样本中相邻成交间隔 p50≈{p50:.1f}s、p90≈{p90:.1f}s、min≈{min_delta}；同秒簇占比(近似)≈{clusters}.",
        )

    return {"conclusion": conclusion, "bullets": bullets}

