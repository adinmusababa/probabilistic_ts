"""Training offline baseline: AdamW, early stopping pada NLL validasi, checkpoint terbaik."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import transformers
from torch.utils.data import DataLoader

from src.models.probabilistic_transformer import ProbabilisticTransformer
from src.utils.seed import set_seed


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


@torch.no_grad()
def evaluate_loss(model: ProbabilisticTransformer, loader: DataLoader, device: torch.device) -> float:
    """Rata-rata NLL (dibobot jumlah sampel) pada sebuah loader."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        loss = model.training_loss(batch)
        bs = batch["past_values"].shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def load_checkpoint(path: str | Path, device: str = "cpu") -> ProbabilisticTransformer:
    ckpt = torch.load(path, map_location=device)
    model = ProbabilisticTransformer(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def train(
    model: ProbabilisticTransformer,
    data,                      # PreparedData
    cfg: dict,                 # config training
    out_dir: str | Path,
    model_cfg: dict,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    gen = torch.Generator()
    gen.manual_seed(cfg["seed"])
    nw = cfg.get("num_workers", 0)
    train_loader = DataLoader(
        data.datasets["train"], batch_size=cfg["batch_size"], shuffle=True,
        generator=gen, num_workers=nw,
    )
    val_loader = DataLoader(
        data.datasets["val"], batch_size=cfg["batch_size"], shuffle=False, num_workers=nw,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0)
    )
    clip = cfg.get("grad_clip")
    min_delta = cfg.get("min_delta", 0.0)

    best, best_epoch, bad = float("inf"), 0, 0
    history: list[dict] = []
    t_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = model.training_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss tidak finite pada epoch {epoch}.")
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            bs = batch["past_values"].shape[0]
            total += loss.item() * bs
            n += bs

        train_loss = total / max(n, 1)
        val_loss = evaluate_loss(model, val_loader, device)
        history.append(
            {"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss,
             "seconds": round(time.time() - t0, 2)}
        )
        print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best - min_delta:
            best, best_epoch, bad = val_loss, epoch, 0
            torch.save(
                {"model_state": model.state_dict(), "model_cfg": model_cfg,
                 "epoch": epoch, "val_nll": val_loss},
                out_dir / "best.pt",
            )
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"early stopping pada epoch {epoch} (terbaik: epoch {best_epoch})")
                break

    pl.DataFrame(history).write_csv(out_dir / "history.csv")
    manifest = {
        "train_cfg": cfg,
        "model_cfg": model_cfg,
        "data": data.meta,
        "best_val_nll": best,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "duration_seconds": round(time.time() - t_start, 1),
        "device": str(device),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "polars": pl.__version__,
        },
        "git_hash": _git_hash(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest