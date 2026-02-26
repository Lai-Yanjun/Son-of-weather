from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    src = root / "src"
    sys.path.insert(0, str(src))

    from polymarket_verify.cli import main as cli_main  # noqa: WPS433 (runtime import)

    return int(cli_main())


if __name__ == "__main__":
    raise SystemExit(main())

