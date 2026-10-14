from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

from src.investigation.safe_staging import atomic_write_bytes


_MAX_AUTH_BYTES = 1024 * 1024
_MINIMUM_TOKEN_LIFETIME_SECONDS = 73 * 60 * 60
_NON_REFRESHABLE_TOKEN = "agentic-scientist-access-token-only"
_NATIVE_AUTH_PATHS = {
    "pi": (".pi/agent/auth.json", "openai-codex"),
    "opencode": (".local/share/opencode/auth.json", "openai"),
}


class SubscriptionAuthError(ValueError):
    pass


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := os.read(descriptor, min(64 * 1024, _MAX_AUTH_BYTES + 1 - total)):
        chunks.append(chunk)
        total += len(chunk)
        if total > _MAX_AUTH_BYTES:
            raise SubscriptionAuthError("Codex auth file is too large")
    return b"".join(chunks)


def _read_private_file(path: Path) -> bytes:
    absolute = path.absolute()
    parent_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    file_fd = -1
    try:
        for part in absolute.parts[1:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            os.close(parent_fd)
            parent_fd = next_fd
        file_fd = os.open(
            absolute.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise SubscriptionAuthError("Codex auth must be a safe regular file owned by this user")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise SubscriptionAuthError("Codex auth file must be private")
        if metadata.st_size > _MAX_AUTH_BYTES:
            raise SubscriptionAuthError("Codex auth file is too large")
        return _read_all(file_fd)
    except SubscriptionAuthError:
        raise
    except OSError as error:
        raise SubscriptionAuthError("Codex auth must be a safe regular file") from error
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


def _json_object(content: bytes | str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SubscriptionAuthError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise SubscriptionAuthError(f"{label} must contain a JSON object")
    return value


def _jwt_claims(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        raise SubscriptionAuthError("Codex access token is malformed")
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(payload)
    except (ValueError, binascii.Error) as error:
        raise SubscriptionAuthError("Codex access token is malformed") from error
    return _json_object(decoded, "Codex access token")


def _codex_oauth(path: Path) -> dict[str, str | int]:
    payload = _json_object(_read_private_file(path), "Codex auth file")
    if payload.get("auth_mode") != "chatgpt":
        raise SubscriptionAuthError("Codex must be signed in with ChatGPT")
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        raise SubscriptionAuthError("Codex ChatGPT tokens are missing")
    access = tokens.get("access_token")
    if not isinstance(access, str) or not access:
        raise SubscriptionAuthError("Codex ChatGPT access token is missing")
    claims = _jwt_claims(access)
    expires = claims.get("exp")
    if isinstance(expires, bool) or not isinstance(expires, int) or expires <= 0:
        raise SubscriptionAuthError("Codex access token expiry is missing")
    required_expiry_raw = os.environ.get("AGENT_CHATGPT_SUBSCRIPTION_REQUIRED_EXPIRY")
    if required_expiry_raw is None:
        required_expiry = int(time.time()) + _MINIMUM_TOKEN_LIFETIME_SECONDS
    else:
        try:
            required_expiry = int(required_expiry_raw)
        except ValueError as error:
            raise SubscriptionAuthError("subscription expiry requirement is invalid") from error
        if required_expiry <= 0:
            raise SubscriptionAuthError("subscription expiry requirement is invalid")
    if expires < required_expiry:
        raise SubscriptionAuthError(
            "Codex ChatGPT access token will not cover the matrix job time limit; "
            "run codex login again before launching"
        )
    claim_auth = claims.get("https://api.openai.com/auth")
    claim_account = claim_auth.get("chatgpt_account_id") if isinstance(claim_auth, dict) else None
    account_id = tokens.get("account_id") or claims.get("chatgpt_account_id") or claim_account
    if not isinstance(account_id, str) or not account_id:
        raise SubscriptionAuthError("Codex ChatGPT account ID is missing")
    return {
        "type": "oauth",
        "refresh": _NON_REFRESHABLE_TOKEN,
        "access": access,
        "expires": expires * 1000,
        "accountId": account_id,
    }


def _harden_auth_directories(home: Path, relative_path: str) -> None:
    descriptor = os.open(home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(descriptor, 0o700)
        for part in Path(relative_path).parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
            os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)


def stage_subscription_auth(agent: str, codex_auth: Path, home: Path) -> None:
    try:
        relative_path, provider = _NATIVE_AUTH_PATHS[agent]
    except KeyError as error:
        raise SubscriptionAuthError(f"unsupported subscription agent: {agent}") from error

    oauth = _codex_oauth(codex_auth)
    native = {provider: oauth}
    content = (json.dumps(native, indent=2, sort_keys=True) + "\n").encode()
    atomic_write_bytes(home, relative_path, content, mode=0o600)
    _harden_auth_directories(home, relative_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=sorted(_NATIVE_AUTH_PATHS), required=True)
    parser.add_argument("--codex-auth", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    try:
        stage_subscription_auth(arguments.agent, arguments.codex_auth, arguments.home)
    except SubscriptionAuthError as error:
        print(f"subscription auth setup failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
