from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


def _key(name: str, labels: dict[str, str] | None = None) -> tuple[str, tuple[tuple[str, str], ...]]:
    return name, tuple(sorted((labels or {}).items()))


@dataclass
class _TimerStat:
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0


class MetricsRegistry:
    """Registro em memória, thread-safe e sem dependências externas.

    Mantém apenas labels de baixa cardinalidade. IDs de paciente/job nunca devem
    ser usados como label.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._started = time.monotonic()
        self._counters: dict[tuple, float] = defaultdict(float)
        self._gauges: dict[tuple, float] = defaultdict(float)
        self._timers: dict[tuple, _TimerStat] = defaultdict(_TimerStat)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[_key(name, labels)] += value

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._gauges[_key(name, labels)] = float(value)

    def inc_gauge(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._gauges[_key(name, labels)] += value

    def observe(self, name: str, seconds: float, **labels: str) -> None:
        with self._lock:
            stat = self._timers[_key(name, labels)]
            stat.count += 1
            stat.total += float(seconds)
            stat.maximum = max(stat.maximum, float(seconds))

    def get_gauge(self, name: str, **labels: str) -> float:
        with self._lock:
            return float(self._gauges.get(_key(name, labels), 0.0))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = [
                {"name": name, "labels": dict(labels), "value": value}
                for (name, labels), value in sorted(self._counters.items())
            ]
            gauges = [
                {"name": name, "labels": dict(labels), "value": value}
                for (name, labels), value in sorted(self._gauges.items())
            ]
            timers = [
                {
                    "name": name,
                    "labels": dict(labels),
                    "count": stat.count,
                    "sum_seconds": stat.total,
                    "max_seconds": stat.maximum,
                    "avg_seconds": (stat.total / stat.count) if stat.count else 0.0,
                }
                for (name, labels), stat in sorted(self._timers.items())
            ]
            return {
                "uptime_seconds": round(time.monotonic() - self._started, 3),
                "counters": counters,
                "gauges": gauges,
                "timers": timers,
            }

    @staticmethod
    def _escape_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def to_prometheus(self) -> str:
        """Exporta formato textual compatível com Prometheus/OpenMetrics básico."""
        with self._lock:
            lines: list[str] = []
            lines.append("# HELP physioai_uptime_seconds Process uptime in seconds")
            lines.append("# TYPE physioai_uptime_seconds gauge")
            lines.append(f"physioai_uptime_seconds {time.monotonic() - self._started:.6f}")

            for (name, labels), value in sorted(self._counters.items()):
                label_text = self._format_labels(labels)
                lines.append(f"{name}{label_text} {value:.6f}")
            for (name, labels), value in sorted(self._gauges.items()):
                label_text = self._format_labels(labels)
                lines.append(f"{name}{label_text} {value:.6f}")
            for (name, labels), stat in sorted(self._timers.items()):
                label_text = self._format_labels(labels)
                lines.append(f"{name}_count{label_text} {stat.count}")
                lines.append(f"{name}_sum_seconds{label_text} {stat.total:.6f}")
                lines.append(f"{name}_max_seconds{label_text} {stat.maximum:.6f}")
            return "\n".join(lines) + "\n"

    def _format_labels(self, labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return ""
        content = ",".join(f'{k}="{self._escape_label(v)}"' for k, v in labels)
        return "{" + content + "}"


metrics = MetricsRegistry()
