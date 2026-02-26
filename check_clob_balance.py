from __future__ import annotations

"""
只读检查：验证 FUNDER_ADDRESS 是否有资金 & L2 凭证是否可用（不下单）。

运行：
  python check_clob_balance.py

输出：
- signer address（由 PRIVATE_KEY 推导）
- funder address（.env 填的）
- collateral(USDC) balance/allowance
"""

import json
import os
from typing import Any, Optional

from dotenv import load_dotenv
from eth_account import Account

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, AssetType, BalanceAllowanceParams


def _mask(s: Optional[str], keep: int = 6) -> str:
    if not s:
        return "(empty)"
    s = str(s)
    if len(s) <= keep:
        return "***"
    return s[:2] + "***" + s[-keep:]


def _as_dict(x: Any) -> Any:
    # 尽量把返回对象变成可打印结构
    if isinstance(x, dict):
        return x
    if hasattr(x, "__dict__"):
        return dict(x.__dict__)
    return x


def main() -> int:
    load_dotenv(".env")

    pk = os.getenv("PRIVATE_KEY")
    if not pk:
        print("ERROR: missing PRIVATE_KEY in .env")
        return 2

    signer = Account.from_key(pk).address
    sig_type = int(os.getenv("SIGNATURE_TYPE", "0"))
    funder = os.getenv("FUNDER_ADDRESS") or ""

    api_key = os.getenv("POLY_API_KEY") or ""
    api_secret = os.getenv("POLY_SECRET") or ""
    api_pass = os.getenv("POLY_PASSPHRASE") or ""

    creds = None
    if api_key and api_secret and api_pass:
        creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_pass)

    print("signer_address:", signer)
    print("signature_type:", sig_type)
    print("funder_address:", funder or "(empty)")
    print("api_key_present:", bool(api_key), "api_key_masked:", _mask(api_key))

    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key=pk,
        creds=creds,
        signature_type=sig_type,
        funder=(funder or None),
    )

    if creds is None:
        print("NOTE: POLY_* creds missing, deriving via L1 (will NOT place orders).")
        derived = client.create_or_derive_api_creds()
        client.set_api_creds(derived)
        # 只展示 apiKey，不展示 secret/passphrase
        derived_dict = {"apiKey": getattr(derived, "api_key", None) or derived.get("apiKey")}
        print("derived_api_key_masked:", _mask(derived_dict.get("apiKey")))

    params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, token_id=None, signature_type=sig_type)
    bal = client.get_balance_allowance(params)
    print("\ncollateral_balance_allowance:")
    print(json.dumps(_as_dict(bal), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

