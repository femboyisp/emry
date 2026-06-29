"""Real-PyTorch training loop instrumented with Emry.

A tiny regression on synthetic data (no dataset download) — the point is the
instrumentation, not the model. Note that `loss` is passed straight to
`run.emit` as a live tensor: no `.item()`, no `.detach()`.

    pip install emry torch
    python examples/torch_train.py

Then watch it (in another shell): `emry watch ./logs/<run-dir>`
or `emry web --run-dir ./logs/<run-dir>`.
"""

from __future__ import annotations

import emry
import torch


def main() -> None:
    torch.manual_seed(0)
    # Synthetic linear target with noise.
    true_w = torch.randn(8, 1)
    x = torch.randn(4096, 8)
    y = x @ true_w + 0.1 * torch.randn(4096, 1)

    model = torch.nn.Linear(8, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)
    loss_fn = torch.nn.MSELoss()

    with emry.run("torch-regression", config={"lr": 1e-2}, metrics=["loss", "lr"]) as run:
        for step in run.steps(500):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            # Hand Emry the tensors directly — coercion takes care of the rest.
            run.emit(loss=loss, lr=torch.tensor(sched.get_last_lr()[0]))
            if step % 50 == 0:
                print(f"step {step:4d}  loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
