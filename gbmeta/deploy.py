"""Deployment cost: latency, throughput, model size, and GPU energy.

Turns a trained model into the numbers an operator needs before deploying it.

Measurement rules that make the numbers defensible:

* **Warm up, then measure.** The first call allocates buffers, JITs kernels and
  faults in pages; including it turns a 30 us model into a 30 ms one.
* **Synchronise CUDA.** ``time.perf_counter`` around an async kernel launch
  measures the launch queue. Every GPU timing here brackets a
  ``torch.cuda.synchronize()``.
* **Report percentiles, not the mean.** A single 200 ms stall from the OS
  scheduler moves the mean and tells a reader nothing; p50/p95/p99 is what an
  operator can plan against.
* **Latency and throughput are different numbers.** Latency is batch-1 p50.
  Peak throughput comes from a batch sweep and is achieved at a much worse
  per-request latency. Reporting one as the other is the most common error in
  this table.
* **Do not invent embedded numbers.** No calibrated Cortex-A72-vs-x86 ratio
  exists for tree ensembles, so this module measures a *single-threaded CPU*
  configuration as a stated proxy and refuses to extrapolate to a Raspberry Pi.
"""
from __future__ import annotations

import gzip
import os
import pickle
import statistics
import time
from dataclasses import dataclass, field

import numpy as np

from .utils import LOG, optional_import

torch = optional_import("torch")

#: Batch sizes used for the throughput sweep.
BATCH_SWEEP = (1, 8, 32, 128, 512, 2048)


def pin_single_thread() -> dict:
    """Force single-threaded execution for the edge-proxy measurement.

    Environment variables must be set **before** OpenMP and BLAS load, which in
    practice means before importing numpy/torch/onnxruntime. When this is called
    later, the env part is a no-op and only the runtime setters take effect --
    so the return value records what actually applied rather than what was asked
    for.
    """
    applied = {}
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        applied[var] = os.environ.get(var)
    if torch is not None:
        try:
            torch.set_num_threads(1)
            applied["torch_num_threads"] = torch.get_num_threads()
        except Exception as exc:  # pragma: no cover
            applied["torch_num_threads_error"] = str(exc)
    applied["note"] = (
        "OMP/MKL thread counts are read at library load time; set them before "
        "importing numpy/torch for the setting to take full effect."
    )
    return applied


# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------
@dataclass
class LatencyResult:
    batch_size: int
    n_reps: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    per_sample_us: float
    throughput_sps: float

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _sync():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency(
    predict_fn, X: np.ndarray, batch_size: int = 1, n_reps: int = 200,
    n_warmup: int = 20, seed: int = 0,
) -> LatencyResult:
    """Time ``predict_fn`` on random batches drawn from ``X``.

    Batches are resampled each repetition so a cached tree path or a warm cache
    line does not flatter the result.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    batch_size = min(batch_size, n)

    def _batch():
        idx = rng.integers(0, n, size=batch_size)
        return np.ascontiguousarray(X[idx], dtype=np.float32)

    for _ in range(n_warmup):
        predict_fn(_batch())
    _sync()

    times = []
    for _ in range(n_reps):
        xb = _batch()
        _sync()
        t0 = time.perf_counter_ns()
        predict_fn(xb)
        _sync()
        times.append((time.perf_counter_ns() - t0) / 1e6)  # ms

    times.sort()
    p50 = statistics.median(times)
    return LatencyResult(
        batch_size=batch_size, n_reps=n_reps,
        p50_ms=p50,
        p95_ms=times[min(int(0.95 * len(times)), len(times) - 1)],
        p99_ms=times[min(int(0.99 * len(times)), len(times) - 1)],
        mean_ms=float(np.mean(times)), min_ms=times[0],
        # Derived from the MEDIAN: the mean is contaminated by scheduler stalls.
        per_sample_us=p50 * 1000.0 / batch_size,
        throughput_sps=batch_size / (p50 / 1000.0),
    )


def latency_sweep(predict_fn, X, batches=BATCH_SWEEP, n_reps: int = 100, seed: int = 0) -> dict:
    """Latency across batch sizes, plus the peak-throughput batch."""
    rows = [measure_latency(predict_fn, X, b, n_reps=n_reps, seed=seed).as_dict() for b in batches]
    best = max(rows, key=lambda r: r["throughput_sps"])
    return {
        "sweep": rows,
        "latency_batch1_p50_ms": rows[0]["p50_ms"],
        "latency_batch1_p99_ms": rows[0]["p99_ms"],
        "peak_throughput_sps": best["throughput_sps"],
        "peak_throughput_batch": best["batch_size"],
        "per_sample_us_at_peak": best["per_sample_us"],
    }


# --------------------------------------------------------------------------
# Model size
# --------------------------------------------------------------------------
def model_footprint(model) -> dict:
    """Serialised size and structural complexity of a fitted model."""
    out = dict(model.complexity() or {})
    try:
        payload = pickle.dumps(model._picklable_payload(), protocol=pickle.HIGHEST_PROTOCOL)
        out["raw_bytes"] = len(payload)
        out["gzip_bytes"] = len(gzip.compress(payload, 6))
        out["raw_mb"] = round(len(payload) / 1e6, 3)
        out["gzip_mb"] = round(out["gzip_bytes"] / 1e6, 3)
    except Exception as exc:  # pragma: no cover
        LOG.warning("could not serialise %s: %s", getattr(model, "name", model), exc)
        out["raw_bytes"] = out["gzip_bytes"] = None
    return out


# --------------------------------------------------------------------------
# Energy
# --------------------------------------------------------------------------
class GpuEnergyMeter:
    """GPU energy over a workload, using NVML's millijoule counter.

    ``nvmlDeviceGetTotalEnergyConsumption`` is a hardware *integrator* available
    from Volta onwards, so it works on a T4 and is far more trustworthy than
    polling ``nvmlDeviceGetPowerUsage`` -- that sensor is an averaged sample, and
    published measurements find only ~25% of a run's wall time is actually
    covered by its samples.

    Two caveats are reported alongside the number rather than buried: the
    counter is *device-wide* (a shared Colab GPU would attribute another
    tenant's work to you), and a measurement window under ~30 s is dominated by
    counter granularity.
    """

    def __init__(self) -> None:
        self.nvml = optional_import("pynvml")
        self.handle = None
        self.available = False
        self.reason = None
        if self.nvml is None:
            self.reason = "nvidia-ml-py not installed (pip install nvidia-ml-py)"
            return
        try:
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.available = True
        except Exception as exc:
            self.reason = f"NVML unavailable: {exc}"

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._e0 = (self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                    if self.available else None)
        self.result = {}
        return self

    def __exit__(self, *exc):
        secs = time.perf_counter() - self._t0
        self.result = {"seconds": secs, "available": self.available, "reason": self.reason}
        if self.available:
            mj = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) - self._e0
            self.result.update({
                "energy_joules": mj / 1000.0,
                "average_power_w": (mj / 1000.0) / max(secs, 1e-9),
                "counter": "nvmlDeviceGetTotalEnergyConsumption (hardware integrator)",
                "device_wide": True,
                "window_adequate": secs >= 30.0,
            })
            if secs < 30.0:
                LOG.warning("energy window only %.1fs - too short to be meaningful (want >=30s)", secs)


def measure_energy(predict_fn, X, min_seconds: float = 30.0, batch_size: int = 512,
                   seed: int = 0) -> dict:
    """Joules per 1000 inferences, measured over a window of at least 30 s.

    CPU energy is deliberately absent: ``/sys/class/powercap`` (RAPL) is not
    readable inside a Colab VM, so any CPU figure would be a TDP guess dressed
    up as a measurement. The table says "not measurable in this environment".
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    batch_size = min(batch_size, n)
    predict_fn(np.ascontiguousarray(X[:batch_size], dtype=np.float32))  # warm up

    meter = GpuEnergyMeter()
    n_samples = 0
    with meter:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < min_seconds:
            idx = rng.integers(0, n, size=batch_size)
            predict_fn(np.ascontiguousarray(X[idx], dtype=np.float32))
            n_samples += batch_size
        _sync()

    out = dict(meter.result)
    out["samples_inferred"] = n_samples
    if out.get("energy_joules") is not None:
        out["joules_per_1000_inferences"] = out["energy_joules"] / max(n_samples, 1) * 1000
    out["cpu_energy"] = (
        "not measurable: RAPL/powercap is not readable inside a Colab VM "
        "(kernel hardening after CVE-2020-8694). No CPU energy is reported "
        "rather than a TDP-based estimate."
    )
    return out


# --------------------------------------------------------------------------
# ONNX export (optional) -- the edge-deployment proxy
# --------------------------------------------------------------------------
def export_onnx(model, n_features: int, out_path, zipmap: bool = False) -> dict:
    """Export a fitted tree/linear model to ONNX.

    ``zipmap=False`` is not cosmetic: the default ZipMap output turns each
    prediction into a Python dict and costs roughly 3x the model's own runtime,
    so leaving it on benchmarks dict construction instead of the classifier.

    Inputs are declared ``FloatTensorType``. Tree converters compare float32
    thresholds against float64 training values, and the resulting threshold
    flips are a documented source of large prediction mismatches -- so the
    caller is expected to verify agreement with :func:`verify_onnx_agreement`.
    """
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    name = getattr(model, "name", "model")
    inner = getattr(model, "model", None)
    if inner is None:
        return {"exported": False, "reason": f"{name}: no sklearn-compatible inner estimator"}

    skl2onnx = optional_import("skl2onnx")
    onnxmltools = optional_import("onnxmltools")
    try:
        if name == "lightgbm" and onnxmltools is not None:
            from onnxmltools.convert.common.data_types import FloatTensorType as FTT
            onx = onnxmltools.convert_lightgbm(
                inner, initial_types=[("input", FTT([None, n_features]))], zipmap=zipmap)
        elif name == "xgboost" and onnxmltools is not None:
            from onnxmltools.convert.common.data_types import FloatTensorType as FTT
            onx = onnxmltools.convert_xgboost(
                inner, initial_types=[("input", FTT([None, n_features]))])
        elif name == "catboost":
            inner.save_model(str(out_path), format="onnx")
            return {"exported": True, "path": str(out_path), "converter": "catboost native",
                    "bytes": out_path.stat().st_size,
                    "caveat": "CatBoost ONNX export does not support categorical features and "
                              "has a documented binary-classification label bug"}
        elif skl2onnx is not None:
            from skl2onnx.common.data_types import FloatTensorType as FTT
            onx = skl2onnx.to_onnx(
                inner, initial_types=[("input", FTT([None, n_features]))],
                options={id(inner): {"zipmap": zipmap}})
        else:
            return {"exported": False, "reason": "no converter available for " + name}
    except Exception as exc:
        LOG.warning("ONNX export failed for %s: %s", name, exc)
        return {"exported": False, "reason": str(exc)}

    out_path.write_bytes(onx.SerializeToString())
    return {"exported": True, "path": str(out_path), "bytes": out_path.stat().st_size,
            "zipmap": zipmap}


def verify_onnx_agreement(onnx_path, model, X: np.ndarray, n: int = 2000) -> dict:
    """Check the exported graph reproduces the original model's predictions.

    Reporting an ONNX latency number without this check is reporting the speed
    of a model whose accuracy was never confirmed.
    """
    ort = optional_import("onnxruntime")
    if ort is None:
        return {"checked": False, "reason": "onnxruntime not installed"}
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    xb = np.ascontiguousarray(X[:n], dtype=np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: xb})
    onnx_pred = np.asarray(out[0]).ravel().astype(int)
    ref_pred = model.predict(xb).astype(int)
    agree = float((onnx_pred == ref_pred).mean())
    return {
        "checked": True, "n_rows": int(len(xb)), "agreement": agree,
        "exact": bool(agree == 1.0),
        "note": None if agree == 1.0 else
        "float32 threshold rounding in the tree converter; report the ONNX latency "
        "only alongside this agreement figure",
    }


def onnx_cpu_latency(onnx_path, X, batch_size: int = 1, n_reps: int = 200,
                     single_thread: bool = True) -> dict:
    """Single-threaded ONNX Runtime latency -- the stated embedded proxy."""
    ort = optional_import("onnxruntime")
    if ort is None:
        return {"measured": False, "reason": "onnxruntime not installed"}
    so = ort.SessionOptions()
    if single_thread:
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    def _predict(xb):
        return sess.run(None, {iname: xb})

    res = measure_latency(_predict, X, batch_size=batch_size, n_reps=n_reps).as_dict()
    res.update({
        "measured": True, "single_thread": single_thread,
        "interpretation": (
            "single-threaded x86 ONNX Runtime. This is a *proxy* for constrained "
            "hardware, not a Raspberry Pi / Jetson measurement. No calibrated "
            "Cortex-A72-vs-x86 ratio exists for tree ensembles, so no "
            "extrapolation to those devices is made."
        ),
    })
    return res


def note_on_int8() -> dict:
    """Why no int8 speed-up is reported for the tree models.

    ``onnxruntime.quantization.quantize_dynamic`` has no entry for
    ``TreeEnsembleClassifier`` in any quantisation registry: it succeeds
    silently and changes nothing. Reporting an int8 speed-up for a GBDT would be
    reporting measurement noise.
    """
    return {
        "applied": False,
        "reason": "ONNX Runtime dynamic quantisation does not cover TreeEnsemble* "
                  "operators; quantize_dynamic is a silent no-op on tree models.",
    }


# --------------------------------------------------------------------------
# Top-level profile
# --------------------------------------------------------------------------
@dataclass
class DeploymentProfile:
    model: str
    latency: dict = field(default_factory=dict)
    footprint: dict = field(default_factory=dict)
    energy: dict = field(default_factory=dict)
    onnx: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def profile_model(
    model, X_test: np.ndarray, name: str | None = None, measure_energy_seconds: float = 0.0,
    onnx_dir=None, seed: int = 0, n_reps: int = 100,
) -> DeploymentProfile:
    """Full deployment profile for one fitted model.

    ``measure_energy_seconds=0`` skips the energy stage; a real measurement
    needs at least 30 s per model, so the notebook runs it only for the models
    that appear in the paper's cost table.
    """
    from .utils import environment_manifest, gpu_name

    name = name or getattr(model, "name", "model")
    prof = DeploymentProfile(model=name)
    LOG.info("profiling %s ...", name)

    prof.latency = latency_sweep(model.predict_proba, X_test, n_reps=n_reps, seed=seed)
    prof.footprint = model_footprint(model)
    prof.environment = {
        "gpu": gpu_name(),
        "uses_gpu": bool(getattr(model, "uses_gpu", False)),
        "cpu_count": os.cpu_count(),
        "packages": environment_manifest()["packages"],
    }
    if measure_energy_seconds > 0:
        prof.energy = measure_energy(model.predict_proba, X_test,
                                     min_seconds=measure_energy_seconds, seed=seed)
    if onnx_dir is not None:
        from pathlib import Path
        p = Path(onnx_dir) / f"{name}.onnx"
        exp = export_onnx(model, X_test.shape[1], p)
        prof.onnx = dict(exp)
        if exp.get("exported"):
            prof.onnx["agreement"] = verify_onnx_agreement(p, model, X_test)
            prof.onnx["cpu_single_thread"] = onnx_cpu_latency(p, X_test, n_reps=n_reps)
            prof.onnx["int8"] = note_on_int8()

    LOG.info("%-16s batch1 p50 %.3f ms | peak %.0f samples/s | %.2f MB gz",
             name, prof.latency["latency_batch1_p50_ms"], prof.latency["peak_throughput_sps"],
             prof.footprint.get("gzip_mb") or float("nan"))
    return prof
