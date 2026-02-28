from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Polymarket 跟单：shadow/live 统一入口（默认 shadow）")
    p.add_argument("--config", default="config.yaml", help="配置文件路径（默认 config.yaml）")
    p.add_argument("--out-dir", default="reports", help="输出目录（默认 reports/）")
    p.add_argument("--state-db", default="shadow_state.db", help="状态数据库（默认 shadow_state.db）")
    p.add_argument("--live-state-db", default=None, help="dual 模式下 live 账本数据库（默认 <state-db>_live）")
    p.add_argument("--days", type=float, default=7.0, help="运行时长（天），默认 7")
    p.add_argument("--live", action="store_true", help="启用实盘小额（默认不下单）")
    p.add_argument("--dual", action="store_true", help="shadow 与 live 同时跑，双账本对比延迟影响（会启用 --live）")
    p.add_argument("--taker", action="store_true", help="允许吃单（风险更高；会覆盖 post-only）")
    p.add_argument("--reset-state", action="store_true", help="启动前删除状态库（state-db 与 live-state-db）")
    p.add_argument("--sync-live-cash", action="store_true", help="启动时强制把 live 账本现金同步为 Data API 余额")
    return p


def main() -> int:
    root = Path(__file__).resolve().parent
    src = root / "src"
    sys.path.insert(0, str(src))

    from polymarket_verify.config_loader import load_config  # noqa: WPS433 (runtime import)
    from polymarket_verify.shadow import run_shadow  # noqa: WPS433 (runtime import)

    args = build_parser().parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    state_db_path = Path(args.state_db)
    live_state_db_path = Path(args.live_state_db) if args.live_state_db else None
    duration_sec = int(float(args.days) * 86400)
    return int(
        run_shadow(
            cfg=cfg,
            out_dir=out_dir,
            state_db_path=state_db_path,
            duration_sec=duration_sec,
            live=bool(args.live),
            taker=bool(args.taker),
            dual=bool(args.dual),
            live_state_db_path=live_state_db_path,
            reset_state=bool(args.reset_state),
            sync_live_cash=bool(args.sync_live_cash),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

