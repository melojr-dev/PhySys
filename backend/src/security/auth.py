from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuthUser:
    username: str
    role: str


class AuthError(ValueError):
    pass


class AuthManager:
    """Autenticação stateless com token HMAC-SHA256.

    Não depende de JWT externo. O token carrega apenas username/role/expiração;
    nunca inclui identificadores clínicos.
    """

    def __init__(
        self,
        *,
        secret: str,
        username: str,
        password: str,
        role: str = "admin",
        ttl_seconds: int = 8 * 60 * 60,
        pseudonym_secret: str | None = None,
    ) -> None:
        if len(secret) < 24:
            raise ValueError("secret de autenticação deve ter ao menos 24 caracteres")
        if not username.strip() or not password:
            raise ValueError("credenciais administrativas inválidas")
        self._secret = secret.encode("utf-8")
        self._username = username.strip()
        self._password = password
        self._role = role.strip() or "admin"
        self._ttl = max(300, int(ttl_seconds))
        self._pseudonym_secret = (pseudonym_secret or secret).encode("utf-8")

    @classmethod
    def from_env(cls) -> "AuthManager":
        env = os.environ.get("PHYSIOAI_ENV", "development").strip().lower()
        secret = os.environ.get("PHYSIOAI_AUTH_SECRET", "")
        password = os.environ.get("PHYSIOAI_ADMIN_PASSWORD", "")
        username = os.environ.get("PHYSIOAI_ADMIN_USER", "admin")
        role = os.environ.get("PHYSIOAI_ADMIN_ROLE", "admin")
        pseudo_secret = os.environ.get("PHYSIOAI_PSEUDONYM_SECRET") or None
        ttl = int(os.environ.get("PHYSIOAI_TOKEN_TTL_SECONDS", str(8 * 60 * 60)))

        if not secret or not password:
            if env == "production":
                raise RuntimeError(
                    "PHYSIOAI_AUTH_SECRET e PHYSIOAI_ADMIN_PASSWORD são obrigatórios em produção"
                )
            # Desenvolvimento local somente. Não é impresso em logs nem persistido.
            secret = secret or ("dev-" + secrets.token_urlsafe(32))
            password = password or "physioai-local"

        if env == "production":
            insecure_markers = (
                "troque-por-",
                "use-outro-segredo-",
                "changeme",
                "change-me",
                "physioai-local",
            )
            values = {
                "PHYSIOAI_AUTH_SECRET": secret,
                "PHYSIOAI_ADMIN_PASSWORD": password,
                "PHYSIOAI_PSEUDONYM_SECRET": pseudo_secret or "",
            }
            weak = [
                name for name, value in values.items()
                if not value or any(marker in value.lower() for marker in insecure_markers)
            ]
            if weak:
                raise RuntimeError(
                    "Segredos de produção ausentes ou ainda com valores de exemplo: "
                    + ", ".join(weak)
                )
        return cls(
            secret=secret,
            username=username,
            password=password,
            role=role,
            ttl_seconds=ttl,
            pseudonym_secret=pseudo_secret,
        )

    @staticmethod
    def _b64encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64decode(data: str) -> bytes:
        padding = "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(data + padding)

    def authenticate(self, username: str, password: str) -> AuthUser:
        ok_user = hmac.compare_digest(username.strip(), self._username)
        ok_password = hmac.compare_digest(password, self._password)
        if not (ok_user and ok_password):
            raise AuthError("credenciais inválidas")
        return AuthUser(username=self._username, role=self._role)

    def issue_token(self, user: AuthUser) -> tuple[str, int]:
        now = int(time.time())
        expires = now + self._ttl
        payload: dict[str, Any] = {
            "sub": user.username,
            "role": user.role,
            "iat": now,
            "exp": expires,
            "nonce": secrets.token_hex(8),
        }
        encoded = self._b64encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        signature = self._b64encode(
            hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        return f"{encoded}.{signature}", expires

    def verify_token(self, token: str) -> AuthUser:
        try:
            encoded, signature = token.split(".", 1)
        except ValueError as exc:
            raise AuthError("token inválido") from exc
        expected = self._b64encode(
            hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(signature, expected):
            raise AuthError("assinatura inválida")
        try:
            payload = json.loads(self._b64decode(encoded))
        except Exception as exc:
            raise AuthError("payload inválido") from exc
        exp = int(payload.get("exp", 0))
        if exp <= int(time.time()):
            raise AuthError("token expirado")
        username = str(payload.get("sub") or "").strip()
        role = str(payload.get("role") or "").strip()
        if not username or not role:
            raise AuthError("token incompleto")
        return AuthUser(username=username, role=role)

    def pseudonymize_patient_id(self, raw_patient_id: str) -> str:
        normalized = " ".join(raw_patient_id.strip().upper().split())
        if not normalized:
            raise ValueError("identificador do paciente não pode ser vazio")
        if normalized.startswith("P-") and len(normalized) == 18 and all(c in "0123456789ABCDEF" for c in normalized[2:]):
            return normalized
        digest = hmac.new(
            self._pseudonym_secret,
            normalized.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest().upper()
        return f"P-{digest[:16]}"
