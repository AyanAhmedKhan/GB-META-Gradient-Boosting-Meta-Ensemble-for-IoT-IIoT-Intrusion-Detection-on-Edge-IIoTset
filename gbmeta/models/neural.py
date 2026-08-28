"""Tabular deep-learning baselines and a controlled trainer.

The v1 paper observed that ResAttDNN and TabTransformer collapsed to ~0.6-0.9%
accuracy and attributed it to a ``WeightedRandomSampler`` x ``OneCycleLR``
interaction. That was a post-hoc explanation of an accident. Here the two
factors are explicit, orthogonal switches (``imbalance`` and ``scheduler``), so
the claim becomes a factorial experiment whose cells can be reported -- which is
what turns a training bug into a finding.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
from sklearn.metrics import f1_score

from ..utils import LOG, optional_import
from .base import BaseLearner, ModelContext, mark_unavailable, register_model

torch = optional_import("torch")
if torch is not None:
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
else:  # pragma: no cover
    nn = None


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------
def _make_mlp(n_features: int, n_classes: int, hidden: int, dropout: float):
    return nn.Sequential(
        nn.Linear(n_features, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, n_classes),
    )


class SEBlock(nn.Module if nn else object):
    """Squeeze-and-Excitation gate over feature channels (Hu et al., CVPR 2018)."""

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        inner = max(dim // reduction, 4)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)

    def forward(self, x):
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))
        return x * s


class ResBlock(nn.Module if nn else object):
    """Two pre-normalised Linear->GELU layers, SE gate, residual skip."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.l1, self.n1 = nn.Linear(dim, dim), nn.LayerNorm(dim)
        self.l2, self.n2 = nn.Linear(dim, dim), nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.se = SEBlock(dim)

    def forward(self, h):
        u = self.drop(torch.nn.functional.gelu(self.n1(self.l1(h))))
        v = self.n2(self.l2(u))
        return torch.nn.functional.gelu(self.se(v) + h)


class ResAttDNN(nn.Module if nn else object):
    """The paper's residual + attention DNN, unchanged in shape."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 256,
                 blocks: int = 3, dropout: float = 0.2, noise: float = 0.01):
        super().__init__()
        self.noise = noise
        self.inp = nn.Sequential(nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Dropout(dropout / 2), nn.Linear(hidden, n_classes))

    def forward(self, x):
        if self.training and self.noise > 0:
            x = x + torch.randn_like(x) * self.noise
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


class TabTransformer(nn.Module if nn else object):
    """Feature-token transformer with a CLS read-out.

    Each scalar feature becomes a ``d``-dim token via a shared projection plus a
    learned per-feature embedding; a CLS token aggregates them. Sequence length
    is ``n_features + 1``, so attention is O(F^2) -- the reason this model needs
    a smaller batch than the others on a T4.
    """

    def __init__(self, n_features: int, n_classes: int, d: int = 32, heads: int = 4,
                 layers: int = 2, ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(1, d)
        self.feat_emb = nn.Parameter(torch.randn(1, n_features, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 64), nn.GELU(), nn.Linear(64, n_classes))

    def forward(self, x):
        tok = self.proj(x.unsqueeze(-1)) + self.feat_emb
        tok = torch.cat([self.cls.expand(tok.size(0), -1, -1), tok], dim=1)
        return self.head(self.enc(tok)[:, 0])


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------
class TorchLearner(BaseLearner):
    """Shared training loop for every neural baseline.

    Knobs that matter for the failure-mode study:

    ``imbalance``
        ``"loss"``    - class weights inside the cross-entropy (default);
        ``"sampler"`` - ``WeightedRandomSampler`` oversampling, unweighted loss;
        ``"none"``    - neither.
    ``scheduler``
        ``"onecycle"``, ``"cosine"`` (with warm-up), ``"plateau"``, ``"flat"``.

    Model selection uses validation **macro-F1**, not accuracy: on a 15-class
    problem where one class holds 72% of the rows, accuracy-based early stopping
    happily selects a model that has given up on every minority class.
    """

    arch = "mlp"
    #: Rows per forward pass at inference time. O(F^2) architectures lower this.
    infer_batch = 8192

    def __init__(self, ctx: ModelContext):
        super().__init__(ctx)
        p = ctx.params
        self.uses_gpu = ctx.device == "cuda"
        self.lr = p.get("lr", 3e-3)
        self.weight_decay = p.get("weight_decay", 1e-4)
        self.epochs = p.get("max_epochs", ctx.budget.max_epochs)
        self.batch_size = p.get("batch_size", ctx.budget.batch_size)
        self.imbalance = p.get("imbalance", "loss")
        self.scheduler_kind = p.get("scheduler", "cosine")
        self.label_smoothing = p.get("label_smoothing", 0.05)
        self.grad_clip = p.get("grad_clip", 1.0)
        self.patience = p.get("patience", max(5, ctx.budget.patience // 3))
        self.warmup_frac = p.get("warmup_frac", 0.1)
        self.history: list = []

    # -- architecture ------------------------------------------------------
    def _build_module(self):
        p, ctx = self.ctx.params, self.ctx
        if self.arch == "mlp":
            return _make_mlp(ctx.n_features, ctx.n_classes,
                             p.get("hidden", 256), p.get("dropout", 0.2))
        if self.arch == "resattdnn":
            return ResAttDNN(ctx.n_features, ctx.n_classes, p.get("hidden", 256),
                             p.get("blocks", 3), p.get("dropout", 0.2), p.get("noise", 0.01))
        if self.arch == "tabtransformer":
            return TabTransformer(ctx.n_features, ctx.n_classes, p.get("d", 32),
                                  p.get("heads", 4), p.get("layers", 2), p.get("ff", 128))
        raise ValueError(self.arch)

    # -- plumbing ----------------------------------------------------------
    def _loader(self, X, y, train: bool):
        ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                           torch.from_numpy(np.asarray(y, dtype=np.int64)))
        sampler = None
        shuffle = train
        if train and self.imbalance == "sampler":
            w = self.ctx.class_weights[y].astype(np.float64)
            sampler = WeightedRandomSampler(
                torch.as_tensor(w, dtype=torch.double), num_samples=len(w), replacement=True,
            )
            shuffle = False
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=0, pin_memory=self.uses_gpu, drop_last=train and len(y) > self.batch_size,
        )

    def _make_scheduler(self, opt, steps_per_epoch: int):
        total = max(1, steps_per_epoch * self.epochs)
        if self.scheduler_kind == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.lr, total_steps=total, pct_start=0.3
            ), "step"
        if self.scheduler_kind == "cosine":
            warm = max(1, int(total * self.warmup_frac))

            def fn(s):
                if s < warm:
                    return (s + 1) / warm
                prog = (s - warm) / max(1, total - warm)
                return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

            return torch.optim.lr_scheduler.LambdaLR(opt, fn), "step"
        if self.scheduler_kind == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=0.5, patience=2
            ), "epoch_metric"
        return None, "none"

    # -- fit ---------------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        t_fit = time.perf_counter()
        dev = torch.device(self.ctx.device)
        torch.manual_seed(self.ctx.seed)
        self.module = self._build_module().to(dev)
        self.n_params = int(sum(p.numel() for p in self.module.parameters()))

        weight = None
        if self.imbalance == "loss" and self.ctx.class_weights is not None:
            weight = torch.as_tensor(self.ctx.class_weights, dtype=torch.float32, device=dev)
        crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)

        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loader = self._loader(X, y, train=True)
        sched, sched_mode = self._make_scheduler(opt, len(loader))

        amp = self.uses_gpu
        scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch, "amp") else None

        best_score, best_state, bad = -np.inf, None, 0

        for epoch in range(self.epochs):
            self.module.train()
            running, seen = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if amp and scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        loss = crit(self.module(xb), yb)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(self.module.parameters(), self.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss = crit(self.module(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.module.parameters(), self.grad_clip)
                    opt.step()
                if sched is not None and sched_mode == "step":
                    sched.step()
                running += float(loss.item()) * len(yb)
                seen += len(yb)

            train_loss = running / max(seen, 1)
            train_acc = self._quick_accuracy(X, y, dev)
            if X_val is not None and len(X_val):
                pv = self._infer(X_val, dev)
                val_acc = float((pv.argmax(1) == y_val).mean())
                val_f1 = float(f1_score(y_val, pv.argmax(1), average="macro", zero_division=0))
            else:
                val_acc = val_f1 = float("nan")

            self.history.append({
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_acc": val_acc, "val_macro_f1": val_f1,
                "lr": float(opt.param_groups[0]["lr"]),
            })
            if sched is not None and sched_mode == "epoch_metric":
                sched.step(val_f1 if np.isfinite(val_f1) else 0.0)

            score = val_f1 if np.isfinite(val_f1) else -train_loss
            if score > best_score + 1e-5:
                best_score, bad = score, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in self.module.state_dict().items()})
            else:
                bad += 1
                if bad >= self.patience:
                    LOG.info("%s: early stop at epoch %d (best macro-F1 %.4f)", self.name, epoch, best_score)
                    break

        if best_state is not None:
            self.module.load_state_dict(best_state)
        self.fit_seconds = time.perf_counter() - t_fit
        self._fitted_classes = np.arange(self.ctx.n_classes)
        self.best_val_macro_f1 = float(best_score)
        return self

    # -- inference ---------------------------------------------------------
    @torch.no_grad() if torch is not None else (lambda f: f)
    def _infer(self, X, dev=None, batch: int = 0) -> np.ndarray:
        dev = dev or torch.device(self.ctx.device)
        batch = batch or self.infer_batch
        self.module.eval()
        out = np.empty((len(X), self.ctx.n_classes), dtype=np.float32)
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i: i + batch], dtype=np.float32)).to(dev)
            logits = self.module(xb).float()
            out[i: i + batch] = torch.softmax(logits, dim=1).cpu().numpy()
        return out

    def _quick_accuracy(self, X, y, dev, n: int = 20_000) -> float:
        idx = slice(0, min(n, len(X)))
        return float((self._infer(X[idx], dev).argmax(1) == y[idx]).mean())

    def predict_proba(self, X):
        return self._check_proba(self._infer(X), len(X))

    def complexity(self):
        return {
            "n_params": getattr(self, "n_params", None),
            "epochs_run": len(self.history),
            "best_val_macro_f1": getattr(self, "best_val_macro_f1", None),
            "imbalance": self.imbalance,
            "scheduler": self.scheduler_kind,
        }

    def _picklable_payload(self):
        return {k: v.detach().cpu().numpy() for k, v in self.module.state_dict().items()}


class MLPLearner(TorchLearner):
    name, arch = "mlp", "mlp"


class ResAttDNNLearner(TorchLearner):
    name, arch = "resattdnn", "resattdnn"


class TabTransformerLearner(TorchLearner):
    name, arch = "tabtransformer", "tabtransformer"
    infer_batch = 1024

    def __init__(self, ctx: ModelContext):
        # O(F^2) attention: quarter the batch so a 100-feature dataset fits a T4.
        super().__init__(ctx)
        self.batch_size = ctx.params.get("batch_size", max(256, ctx.budget.batch_size // 4))
        self.lr = ctx.params.get("lr", 1e-3)


if torch is not None:
    register_model("mlp")(lambda ctx: MLPLearner(ctx))
    register_model("resattdnn")(lambda ctx: ResAttDNNLearner(ctx))
    register_model("tabtransformer")(lambda ctx: TabTransformerLearner(ctx))
else:  # pragma: no cover
    for _k in ("mlp", "resattdnn", "tabtransformer"):
        mark_unavailable(_k, "pip install torch")
