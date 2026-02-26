from __future__ import annotations

"""
只生成/派生 CLOB 的 L2 API creds（不下单）。

用途：
- 你想把 POLY_API_KEY / POLY_SECRET / POLY_PASSPHRASE 填进 .env，
  以便后续实盘脚本走 L2，不必每次都用 L1 去 create/derive。

注意：
- create_api_key 会让“旧 key 失效”；create_or_derive 会优先尝试 derive（默认 nonce=0），
  不存在才会 create。
- 永远不要提交 .env / 私钥 / secret。

运行：
  python gen_clob_api_creds.py
  python gen_clob_api_creds.py --nonce 0
"""

import argparse
import json
import os

from dotenv import load_dotenv
from eth_account import Account
from py_clob_client.client import ClobClient


def _normalize_private_key(pk_raw: str) -> str:
    """
    期望输入为 0x + 64 hex（32 bytes）。
    常见错误：把地址(0x+40hex)当成私钥、复制时缺字符、带引号/空格。
    """
    pk = (pk_raw or "").strip().strip("\"'").strip()
    if pk.startswith("0x") or pk.startswith("0X"):
        pk_body = pk[2:]
    else:
        pk_body = pk

    # 只做格式校验，不输出 key 本体
    if len(pk_body) == 40:
        raise SystemExit(
            "PRIVATE_KEY 看起来是 20 bytes（40位hex）的“地址”，不是私钥。\n"
            "请在 .env 中填写 32 bytes（64位hex）的私钥（可带 0x 前缀）。"
        )
    if len(pk_body) != 64:
        raise SystemExit(
            f"PRIVATE_KEY 长度不对：需要 64 位十六进制字符(=32 bytes)，你现在是 {len(pk_body)}。\n"
            "请确认复制的是“私钥”，不是地址/txhash，也没有缺字符。"
        )
    try:
        int(pk_body, 16)
    except ValueError as e:
        raise SystemExit("PRIVATE_KEY 不是有效的十六进制字符串（应为 0-9a-f）。") from e
    return "0x" + pk_body.lower()


def _creds_to_dict(creds: object) -> dict[str, str]:
    """
    py-clob-client 可能返回 dict 或 ApiCreds 对象。
    统一转换成 {apiKey, secret, passphrase} 便于打印/写入 .env。
    """
    if isinstance(creds, dict):
        api_key = creds.get("apiKey") or creds.get("api_key")
        secret = creds.get("secret") or creds.get("apiSecret") or creds.get("api_secret")
        passphrase = creds.get("passphrase") or creds.get("apiPassphrase") or creds.get("api_passphrase")
        return {
            "apiKey": str(api_key or ""),
            "secret": str(secret or ""),
            "passphrase": str(passphrase or ""),
        }

    # ApiCreds(api_key=..., api_secret=..., api_passphrase=...)
    api_key = getattr(creds, "api_key", None)
    secret = getattr(creds, "api_secret", None)
    passphrase = getattr(creds, "api_passphrase", None)
    if api_key or secret or passphrase:
        return {
            "apiKey": str(api_key or ""),
            "secret": str(secret or ""),
            "passphrase": str(passphrase or ""),
        }

    # fallback
    return {"apiKey": "", "secret": "", "passphrase": ""}


def main() -> int:
    load_dotenv(".env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--nonce", type=int, default=0, help="API key nonce（默认 0）")
    args = ap.parse_args()

    pk_raw = os.getenv("PRIVATE_KEY")
    pk = _normalize_private_key(pk_raw or "")
    if not pk:
        raise SystemExit("缺少 PRIVATE_KEY（请在 .env 填写）")

    sig_type = int(os.getenv("SIGNATURE_TYPE", "0"))
    funder = os.getenv("FUNDER_ADDRESS") or None

    signer_addr = Account.from_key(pk).address
    print("Signer address:", signer_addr)
    print("SIGNATURE_TYPE:", sig_type)
    print("FUNDER_ADDRESS:", funder or "(empty)")
    if sig_type in (1, 2) and not funder:
        print("NOTE: 你选择了 SIGNATURE_TYPE=1/2，但 FUNDER_ADDRESS 为空。")
        print("      生成 API creds 可能仍然成功，但后续下单/查询余额通常需要正确的 FUNDER_ADDRESS。")

    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key=pk,
        signature_type=sig_type,
        funder=funder,
    )

    raw_creds = client.create_or_derive_api_creds(args.nonce)
    creds = _creds_to_dict(raw_creds)
    print("\nAPI creds (save into .env):")
    print(json.dumps(creds, ensure_ascii=False, indent=2))

    print("\n.env lines:")
    print(f"POLY_API_KEY={creds.get('apiKey','')}")
    print(f"POLY_SECRET={creds.get('secret','')}")
    print(f"POLY_PASSPHRASE={creds.get('passphrase','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

