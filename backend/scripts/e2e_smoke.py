#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class SmokeFailure(RuntimeError):
    pass


def http_request(
    url: str,
    *,
    method: str = "GET",
    token: str | None = None,
    data: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 15.0,
) -> tuple[int, bytes, dict[str, str]]:
    headers: dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), {k.lower(): v for k, v in exc.headers.items()}


def request_json(url: str, **kwargs: Any) -> tuple[int, dict[str, Any], dict[str, str]]:
    status, raw, headers = http_request(url, **kwargs)
    if not raw:
        return status, {}, headers
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise SmokeFailure(f"Resposta n�o-JSON em {url}: HTTP {status}") from exc
    if not isinstance(payload, dict):
        raise SmokeFailure(f"Resposta JSON inesperada em {url}")
    return status, payload, headers


def json_body(payload: dict[str, Any]) -> tuple[bytes, str]:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8"), "application/json"


def multipart_body(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = "----PhysioAISmoke" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
            str(value).encode("utf-8"),
            b"\r\n",
        ])
    chunks.extend([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode(),
        b"Content-Type: video/mp4\r\n\r\n",
        file_path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class RawWebSocket:
    def __init__(self, url: str, token: str, timeout: float) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"ws", "wss"}:
            raise SmokeFailure(f"WebSocket scheme inv�lido: {parsed.scheme}")
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        raw = socket.create_connection((host, port), timeout=timeout)
        if parsed.scheme == "wss":
            raw = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
        raw.settimeout(timeout)
        self.sock = raw
        self.buffer = bytearray()
        self.closed = False

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        host_header = host if port in {80, 443} else f"{host}:{port}"
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            f"Sec-WebSocket-Protocol: physioai, {token}\r\n"
            "\r\n"
        ).encode("ascii")
        self.sock.sendall(request)
        headers = self._read_headers()
        first = headers.split("\r\n", 1)[0]
        if " 101 " not in first:
            raise SmokeFailure(f"Handshake WebSocket falhou: {first}")
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        header_map = {}
        for line in headers.split("\r\n")[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                header_map[k.strip().lower()] = v.strip()
        if header_map.get("sec-websocket-accept") != expected:
            raise SmokeFailure("Sec-WebSocket-Accept inv�lido")
        if header_map.get("sec-websocket-protocol") != "physioai":
            raise SmokeFailure("Servidor n�o negociou subprotocolo physioai")

    def _read_headers(self) -> str:
        marker = b"\r\n\r\n"
        while marker not in self.buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise SmokeFailure("Conex�o encerrada durante handshake WebSocket")
            self.buffer.extend(chunk)
        idx = self.buffer.index(marker) + len(marker)
        raw = bytes(self.buffer[:idx])
        del self.buffer[:idx]
        return raw.decode("iso-8859-1")

    def _recv_exact(self, n: int) -> bytes:
        while len(self.buffer) < n:
            chunk = self.sock.recv(max(4096, n - len(self.buffer)))
            if not chunk:
                raise EOFError("WebSocket encerrado")
            self.buffer.extend(chunk)
        out = bytes(self.buffer[:n])
        del self.buffer[:n]
        return out

    def _send_control(self, opcode: int, payload: bytes = b"") -> None:
        mask = os.urandom(4)
        length = len(payload)
        if length > 125:
            payload = payload[:125]
            length = len(payload)
        header = bytes([0x80 | opcode, 0x80 | length])
        masked = bytes(value ^ mask[i % 4] for i, value in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def recv_json(self) -> dict[str, Any] | None:
        while True:
            first, second = self._recv_exact(2)
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if masked else None
            payload = self._recv_exact(length) if length else b""
            if mask:
                payload = bytes(value ^ mask[i % 4] for i, value in enumerate(payload))
            if opcode == 0x8:
                self.closed = True
                return None
            if opcode == 0x9:
                self._send_control(0xA, payload)
                continue
            if opcode != 0x1:
                continue
            decoded = json.loads(payload.decode("utf-8"))
            return decoded if isinstance(decoded, dict) else None

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._send_control(0x8)
        except Exception:
            pass
        try:
            self.sock.close()
        finally:
            self.closed = True


def ws_url(base_url: str, path: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urllib.parse.urlunsplit((scheme, parsed.netloc, path, "", ""))


def require(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise SmokeFailure(f"FAIL {name}: {detail}")
    print(f"PASS  {name:24s} {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description="PhysioAI end-to-end smoke test")
    ap.add_argument("--base-url", default="http://localhost")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", required=True)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--expect-valid-measurement", action="store_true")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    try:
        status, health, headers = request_json(base + "/health")
        require("liveness", status == 200 and health.get("status") == "ok", f"HTTP {status}")
        require("request_id", bool(headers.get("x-request-id")), "header presente")

        status, ready, _ = request_json(base + "/ready")
        require("readiness", status == 200 and ready.get("status") == "ready", str(ready.get("checks")))

        status, _, _ = request_json(base + "/api/auth/me")
        require("auth_required", status == 401, f"HTTP {status}")

        body, ctype = json_body({"username": args.user, "password": args.password})
        status, login, _ = request_json(base + "/api/auth/login", method="POST", data=body, content_type=ctype)
        token = str(login.get("access_token") or "")
        require("login", status == 200 and bool(token), f"HTTP {status}")

        status, me, _ = request_json(base + "/api/auth/me", token=token)
        require("auth_me", status == 200 and me.get("username") == args.user, str(me))

        status, version, _ = request_json(base + "/version")
        require("version", status == 200 and bool(version.get("version")), str(version.get("version")))

        status, movements, _ = request_json(base + "/api/config/movimentos", token=token)
        enabled = movements.get("movements") or movements.get("items") or []
        require("clinical_movements", status == 200 and len(enabled) >= 2, f"count={len(enabled)}")

        status, system, _ = request_json(base + "/api/system/health", token=token)
        require("system_health", status == 200 and system.get("status") in {"ok", "degraded"}, str(system.get("status")))

        status, metrics, _ = request_json(base + "/api/system/metrics", token=token)
        require("metrics", status == 200 and "counters" in metrics, f"uptime={metrics.get('uptime_seconds')}")

        if args.video is None:
            print("SKIP  analysis_pipeline         use --video para testar upload + WS + infer�ncia")
            return 0
        video = args.video.expanduser().resolve()
        require("video_exists", video.is_file() and video.stat().st_size > 0, str(video))

        preflight_fields = {
            "id_paciente": "E2E-SMOKE-PATIENT",
            "movimento": "Flex�o de Cotovelo",
            "lado": "direito",
        }
        body, ctype = multipart_body(preflight_fields, "video_a", video)
        status, preflight, _ = request_json(
            base + "/api/capture/preflight",
            method="POST", token=token, data=body, content_type=ctype,
            timeout=min(args.timeout, 60.0),
        )
        require("capture_preflight", status == 200 and bool(preflight.get("preflight_id")), f"HTTP {status}")
        require("preflight_contract", "checks" in preflight and "score" in preflight, str(preflight.get("status")))

        job_id = str(uuid.uuid4())
        fields = {
            "id_paciente": "E2E-SMOKE-PATIENT",
            "movimento": "Flex�o de Cotovelo",
            "lado": "direito",
            "offset_ms": "0",
            "tipo_sensor": "Nenhum",
            "modo_gpu": "false",
            "retornar_frames": "false",
            "gerar_graficos": "false",
            "retornar_skeleton": "false",
            "job_id": job_id,
            "preflight_id": str(preflight["preflight_id"]),
        }
        if not bool(preflight.get("can_analyze")):
            fields["preflight_override_reason"] = "Smoke E2E controlado: prosseguir apesar da reprova��o do preflight."
        body = urllib.parse.urlencode(fields).encode("utf-8")
        status, accepted, _ = request_json(
            base + "/api/analise/processar",
            method="POST", token=token, data=body, content_type="application/x-www-form-urlencoded",
            timeout=min(args.timeout, 60.0),
        )
        require("analysis_accepted", status == 202 and accepted.get("job_id") == job_id, f"HTTP {status}")

        socket_client = RawWebSocket(ws_url(base, f"/ws/analise/{job_id}"), token, args.timeout)
        frame_events = 0
        progress_events = 0
        valid_measurements = 0
        terminal: dict[str, Any] | None = None
        deadline = time.monotonic() + args.timeout
        try:
            while time.monotonic() < deadline:
                event = socket_client.recv_json()
                if event is None:
                    break
                kind = event.get("type")
                if kind == "analysis.frame":
                    frame_events += 1
                    measurement = event.get("measurement") or {}
                    require(
                        "frame_contract",
                        "angleClinicalDeg" in measurement and "measurementValid" in measurement,
                        f"frame={event.get('frame')}",
                    ) if frame_events == 1 else None
                    valid_measurements += int(bool(measurement.get("measurementValid")))
                elif kind == "progress":
                    progress_events += 1
                elif kind in {"done", "error", "cancelled"}:
                    terminal = event
                    break
        finally:
            socket_client.close()

        require("websocket_frames", frame_events > 0, f"frames={frame_events}, progress={progress_events}")
        require("analysis_terminal", terminal is not None and terminal.get("type") == "done", str(terminal))
        if args.expect_valid_measurement:
            require("valid_measurement", valid_measurements > 0, f"valid={valid_measurements}")

        status, result, _ = request_json(base + f"/api/analise/resultado/{job_id}", token=token, timeout=30)
        require("analysis_result", status == 200 and result.get("status") == "ok", f"HTTP {status}")
        require("result_hash", bool(result.get("result_sha256")), str(result.get("result_sha256", ""))[:16])

        status, session, _ = request_json(base + f"/api/sessoes/{job_id}", token=token)
        session_data = session.get("session") or {}
        require("session_persisted", status == 200 and session_data.get("status") == "completed", str(session_data.get("status")))
        require("measurements_persisted", int(session.get("measurement_count") or 0) == frame_events, f"n={session.get('measurement_count')}")

        print(f"E2E OK job={job_id} frames={frame_events} valid={valid_measurements}")
        return 0
    except (SmokeFailure, urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"E2E ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
