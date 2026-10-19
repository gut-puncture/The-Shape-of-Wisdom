from __future__ import annotations

from run_full_sweep import build_full_sweep_commands
from run_smoke_test import build_smoke_commands


def test_smoke_commands_propagate_device_and_run_id() -> None:
    commands = build_smoke_commands(run_id="smoke_123", device="cuda:0")
    flat = [" ".join(command) for command in commands]
    assert any("--run-id smoke_123" in command for command in flat)
    assert any("--device cuda:0" in command for command in flat if "run_controls.py" in command)


def test_full_sweep_commands_propagate_device_and_run_id() -> None:
    commands = build_full_sweep_commands(run_id="paper_123", device="cuda:1")
    flat = [" ".join(command) for command in commands]
    assert any("--run-id paper_123" in command for command in flat)
    assert any("run_gpu_preflight.py" in command for command in flat)
    assert any("--device cuda:1" in command for command in flat if "run_controls.py" in command)
