from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional


THERMAL_PRESSURE_ORDER = ["nominal", "fair", "serious", "critical"]
_THERMAL_PRESSURE_RANK = {k: i for i, k in enumerate(THERMAL_PRESSURE_ORDER)}


def _normalize_level(level: str) -> str:
    return str(level).strip().lower()


def _rank(level: str) -> int:
    k = _normalize_level(level)
    if k not in _THERMAL_PRESSURE_RANK:
        raise ValueError(f"unknown thermal pressure level: {level!r}")
    return int(_THERMAL_PRESSURE_RANK[k])


def read_powermetrics_thermal_pressure_level(*, timeout_seconds: float = 15.0) -> Optional[str]:
    """
    Best-effort: query macOS thermal pressure level via `powermetrics`.

    Notes:
    - `powermetrics` requires root; we use non-interactive sudo (`sudo -n`).
    - Returns one of: nominal/fair/serious/critical, or None if unavailable.
    """
    argv = ["sudo", "-n", "powermetrics", "-n", "1", "-i", "1000", "-s", "thermal"]
    try:
        out = subprocess.check_output(argv, stderr=subprocess.STDOUT, timeout=float(timeout_seconds))
    except Exception:
        return None
    text = out.decode("utf-8", errors="replace")
    m = re.search(r"Current pressure level:\s*([A-Za-z]+)", text)
    if not m:
        return None
    level = _normalize_level(m.group(1))
    if level not in _THERMAL_PRESSURE_RANK:
        return None
    return level


@dataclass(frozen=True)
class ThermalHygieneConfig:
    enabled: bool
    provider: str
    cutoff_level: str
    cooldown_seconds: int
    check_interval_seconds: int

    @staticmethod
    def from_cfg(cfg: Dict[str, Any] | None) -> "ThermalHygieneConfig":
        if not isinstance(cfg, dict):
            # Back-compat: if a run_config predates thermal hygiene, keep it disabled
            # rather than silently introducing sleeps into historical runs.
            return ThermalHygieneConfig(
                enabled=False,
                provider="powermetrics_thermal_pressure",
                cutoff_level="serious",
                cooldown_seconds=20 * 60,
                check_interval_seconds=30,
            )
        enabled = bool(cfg.get("enabled", True))
        provider = str(cfg.get("provider", "powermetrics_thermal_pressure"))
        cutoff_level = str(cfg.get("cutoff_level", "serious"))
        cooldown_seconds = int(cfg.get("cooldown_seconds", 20 * 60))
        check_interval_seconds = int(cfg.get("check_interval_seconds", 30))
        return ThermalHygieneConfig(
            enabled=enabled,
            provider=provider,
            cutoff_level=cutoff_level,
            cooldown_seconds=cooldown_seconds,
            check_interval_seconds=check_interval_seconds,
        )


class ThermalGovernor:
    """
    Cooperative thermal governor:
    - Inference loops periodically call `maybe_cooldown(...)`.
    - If thermal pressure is at/above the cutoff, the governor sleeps for cooldown_seconds.

    This is intentionally cooperative (not a background thread), so it only runs while
    inference work is actively progressing.
    """

    def __init__(
        self,
        *,
        cfg: ThermalHygieneConfig,
        events_path: Path,
        read_level_fn: Callable[[], Optional[str]] = read_powermetrics_thermal_pressure_level,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cfg = cfg
        self._events_path = events_path
        self._read_level_fn = read_level_fn
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._last_check_ts: Optional[float] = None

        # Validate early (fail-fast).
        if self._cfg.provider != "powermetrics_thermal_pressure":
            raise ValueError(f"unknown thermal provider: {self._cfg.provider!r}")
        _rank(self._cfg.cutoff_level)
        if self._cfg.cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be positive")
        if self._cfg.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be positive")

    @property
    def cfg(self) -> ThermalHygieneConfig:
        return self._cfg

    def _append_event(self, obj: Dict[str, Any]) -> None:
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        obj2 = dict(obj)
        obj2["ts_utc"] = datetime.now(timezone.utc).isoformat()
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj2, ensure_ascii=False, sort_keys=True) + "\n")

    def maybe_cooldown(self, *, stage: str, model_id: str, model_revision: str) -> Dict[str, Any]:
        """
        Returns a small dict describing the action taken (for tests/metrics).
        """
        if not self._cfg.enabled:
            return {"checked": False, "enabled": False}

        now = float(self._time_fn())
        if self._last_check_ts is not None:
            if (now - self._last_check_ts) < float(self._cfg.check_interval_seconds):
                return {"checked": False, "enabled": True}
        self._last_check_ts = now

        level = None
        if self._cfg.provider == "powermetrics_thermal_pressure":
            level = self._read_level_fn()

        self._append_event(
            {
                "event": "thermal_check",
                "stage": stage,
                "model_id": model_id,
                "model_revision": model_revision,
                "provider": self._cfg.provider,
                "level": level,
                "cutoff_level": self._cfg.cutoff_level,
            }
        )

        if level is None:
            return {"checked": True, "enabled": True, "level": None, "cooled_down": False}

        if _rank(level) < _rank(self._cfg.cutoff_level):
            return {"checked": True, "enabled": True, "level": level, "cooled_down": False}

        # Cooldown (cooperative sleep)
        self._append_event(
            {
                "event": "cooldown_start",
                "stage": stage,
                "model_id": model_id,
                "model_revision": model_revision,
                "level": level,
                "cooldown_seconds": int(self._cfg.cooldown_seconds),
            }
        )
        self._sleep_fn(float(self._cfg.cooldown_seconds))
        self._append_event(
            {
                "event": "cooldown_end",
                "stage": stage,
                "model_id": model_id,
                "model_revision": model_revision,
                "level": level,
            }
        )
        return {"checked": True, "enabled": True, "level": level, "cooled_down": True}

