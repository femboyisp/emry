"""End-to-end check with real PyTorch tensors.

Validates the headline claim — pass `loss` straight to `run.emit` without
`.item()` — against an actual training step. Skipped when torch isn't installed
(the default CI matrix); a dedicated CI job installs CPU torch and runs it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import emry  # noqa: E402  (after importorskip)


def test_real_torch_tensors_emit_finite_floats(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = torch.nn.MSELoss()
    x, y = torch.randn(32, 4), torch.randn(32, 1)

    with emry.run(
        "torch-smoke",
        metrics=["loss", "lr"],
        live="off",
        mode="file",
        log_dir=str(tmp_path),
    ) as run:
        for _ in run.steps(5):
            opt.zero_grad()
            loss = loss_fn(model(x), y)  # 0-dim tensor, requires_grad=True
            loss.backward()
            opt.step()
            # Pass the live loss tensor and a 0-dim lr tensor straight in — no
            # `.item()`, no `.detach()`. Coercion handles it.
            run.emit(loss=loss, lr=torch.tensor(opt.param_groups[0]["lr"]))

    run_dir = next(tmp_path.iterdir())
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 5
    for row in rows:
        loss_val = row["values"]["loss"]
        assert isinstance(loss_val, float) and math.isfinite(loss_val)
        assert row["values"]["lr"] == pytest.approx(0.05)
