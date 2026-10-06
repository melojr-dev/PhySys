from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone


_STANDARD = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName", "processName",
    "process", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except TypeError:
                    payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_structured_logging() -> None:
    level = os.environ.get("PHYSIOAI_LOG_LEVEL", "INFO").upper()
    as_json = os.environ.get("PHYSIOAI_LOG_FORMAT", "json").lower() == "json"
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    formatter: logging.Formatter
    if as_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    for handler in root.handlers:
        handler.setFormatter(formatter)
