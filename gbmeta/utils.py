"""Determinism, timing, resource accounting and small IO helpers."""
from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import json
import logging
import os
import platform
import random
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

LOG = logging.getLogger("gbmeta")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    if not LOG.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S"))
        LOG.addHandler(h)
    LOG.setLevel(level)
    LOG.propagate = False
    return LOG


# --------------------------------------------------------------------------
# Optional dependencies
# --------------------------------------------------------------------------
def optional_import(name: str):
    """Import ``name`` or return ``None``.

    The study degrades gracefully: a machine without CatBoost still produces a
    complete (smaller) results table rather than crashing halfway through.
    Every skipped component is recorded in the run manifest.
    """
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - environment dependent
        LOG.debug("optional dependency %s unavailable: %s", name, exc)
        return None


def available_backends() -> "dict[str, str | None]":
    out = {}
    for mod in (
        "numpy", "pandas", "sklearn", "scipy", "statsmodels", "lightgbm",
        "xgboost", "catboost", "torch", "optuna", "matplotlib",
        "onnxruntime", "skl2onnx", "onnxmltools", "pynvml", "river",
        "pytorch_tabnet",
    ):
        m = optional_import(mod)
        out[mod] = getattr(m, "__version__", "unknown") if m is not None else None
    return out


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every RNG that can influence a reported number.

    ``PYTHONHASHSEED`` is set for completeness even though it only takes effect
    in a fresh interpreter -- the run manifest records its value at import time,
    which is what actually matters for exact reproduction.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch = optional_import("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "auto") -> str:
    torch = optional_import("torch")
    if torch is None:
        return "cpu"
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_name():
    torch = optional_import("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


# --------------------------------------------------------------------------
# Timing and memory
# --------------------------------------------------------------------------
class Timer:
    """Monotonic wall-clock timer. ``perf_counter`` -- never ``time.time``."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._t0
        if self.label:
            LOG.info("%s took %.2fs", self.label, self.elapsed)


def peak_rss_gb() -> float:
    psutil = optional_import("psutil")
    if psutil is None:
        return float("nan")
    return psutil.Process().memory_info().rss / 1e9


def system_ram_gb():
    psutil = optional_import("psutil")
    if psutil is None:
        return (float("nan"), float("nan"))
    vm = psutil.virtual_memory()
    return (vm.used / 1e9, vm.total / 1e9)


def vram_gb():
    torch = optional_import("torch")
    if torch is None or not torch.cuda.is_available():
        return (0.0, 0.0)
    used = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return (used, total)


def purge() -> None:
    """Free Python and CUDA memory between pipeline stages."""
    gc.collect()
    torch = optional_import("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@contextlib.contextmanager
def resource_probe(label: str) -> Iterator[dict]:
    """Record wall time, peak RSS and peak VRAM around a block of work."""
    purge()
    rec = {"label": label}
    rss0 = peak_rss_gb()
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        rec["seconds"] = time.perf_counter() - t0
        rec["rss_gb"] = peak_rss_gb()
        rec["rss_delta_gb"] = rec["rss_gb"] - rss0
        rec["vram_peak_gb"] = vram_gb()[0]
        LOG.info(
            "%-24s %6.1fs  RSS %.2f GB (+%.2f)  VRAM %.2f GB",
            label, rec["seconds"], rec["rss_gb"], rec["rss_delta_gb"], rec["vram_peak_gb"],
        )


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
class _NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        return super().default(o)


def write_json(path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, cls=_NpEncoder), encoding="utf-8")
    return path


def read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def environment_manifest() -> dict:
    """Everything needed to reproduce a number, captured at run time."""
    _used, total = system_ram_gb()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(total, 2),
        "gpu": gpu_name(),
        "vram_total_gb": round(vram_gb()[1], 2),
        "packages": available_backends(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
