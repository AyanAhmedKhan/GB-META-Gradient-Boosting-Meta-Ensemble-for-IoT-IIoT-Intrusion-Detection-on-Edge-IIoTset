"""Download exactly the files the study needs -- and nothing else.

``kagglehub.dataset_download(handle)`` without ``path=`` pulls the *whole*
archive. For Edge-IIoTset that is a 1.7 GB zip expanding to 10.5 GB, which fills
a Colab disk before training starts. Every entry below therefore names the single
file inside the archive, which Kaggle serves individually.

Usage::

    python -m gbmeta.datasets.fetch --all          # everything the study needs
    python -m gbmeta.datasets.fetch edge_iiotset   # one dataset
    python -m gbmeta.datasets.fetch --check        # what is already on disk

No Kaggle credentials are required: all six datasets are public and need no
consent click. If a token is ever needed, ``kagglehub.login()`` or a Colab
secret named ``KAGGLE_API_TOKEN`` both work.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ..config import DATA_DIR
from ..utils import LOG, optional_import, setup_logging, sha256_file

#: dataset key -> (kaggle handle, [(path inside archive, local filename)])
SOURCES = {
    "edge_iiotset": (
        "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
        [("Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv",
          "ML-EdgeIIoT-dataset.csv")],
    ),
    "edge_iiotset_full": (
        "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
        [("Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv",
          "DNN-EdgeIIoT-dataset.csv")],
    ),
    "nslkdd": (
        "hassan06/nslkdd",
        [("KDDTrain+.txt", "KDDTrain+.txt"), ("KDDTest+.txt", "KDDTest+.txt")],
    ),
    "unsw_nb15": (
        "mrwellsdavid/unsw-nb15",
        [("UNSW_NB15_testing-set.csv", "UNSW_NB15_testing-set.csv"),
         ("UNSW_NB15_training-set.csv", "UNSW_NB15_training-set.csv")],
    ),
    "ton_iot": (
        "alaaelmor/ton-iot-train-test-network",
        [("Train_Test_Network.csv", "Train_Test_Network.csv")],
    ),
    "cicids2017": (
        "chethuhn/network-intrusion-dataset",
        [(f, f) for f in (
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        )],
    ),
    "botiot": (
        "liuwoo/botiot-2018",
        [("UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv",
          "UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv")],
    ),
}

#: Approximate uncompressed size of each dataset's required files, so the user
#: knows what they are committing to before the download starts.
APPROX_MB = {
    "edge_iiotset": 78, "edge_iiotset_full": 1161, "nslkdd": 22,
    "unsw_nb15": 46, "ton_iot": 67, "cicids2017": 470, "botiot": 88,
}

DEFAULT_KEYS = ("edge_iiotset", "nslkdd", "unsw_nb15", "ton_iot", "cicids2017")


def target_dir(key: str, data_dir=None) -> Path:
    # The *_full variant lives beside the base dataset.
    base = key.replace("_full", "")
    d = Path(data_dir or DATA_DIR) / base
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_direct(handle: str, inner: str, dest: Path) -> Path:
    """Fetch one file through Kaggle's public per-file endpoint.

    ``kagglehub`` 1.x requires Python >= 3.10 and is not installed everywhere.
    The REST endpoint below needs no credentials for public datasets and returns
    a zip containing the single requested file, which keeps the fallback as
    cheap as the happy path.
    """
    import io
    import urllib.parse
    import urllib.request
    import zipfile

    url = ("https://www.kaggle.com/api/v1/datasets/download/"
           f"{handle}?file_name={urllib.parse.quote(inner)}")
    req = urllib.request.Request(url, headers={"User-Agent": "gbmeta-v2/2.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = resp.read()

    dest.parent.mkdir(parents=True, exist_ok=True)
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            member = next((m for m in zf.namelist() if m.endswith(Path(inner).name)), None)
            if member is None:  # pragma: no cover
                raise FileNotFoundError(f"{inner} not present in the returned archive")
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    else:
        # Some slugs serve the raw CSV rather than a zip.
        dest.write_bytes(payload)
    return dest


def fetch(key: str, data_dir=None, force: bool = False) -> list:
    if key not in SOURCES:
        raise KeyError(f"unknown dataset {key!r}; available: {sorted(SOURCES)}")
    kagglehub = optional_import("kagglehub")
    handle, files = SOURCES[key]
    if kagglehub is None:
        LOG.info("kagglehub not installed - using the public Kaggle file endpoint")
    dest_dir = target_dir(key, data_dir)
    written = []
    for inner, name in files:
        dest = dest_dir / name
        if dest.exists() and not force:
            LOG.info("%-18s %s already present (%.1f MB)", key, name, dest.stat().st_size / 1e6)
            written.append(dest)
            continue
        LOG.info("%-18s downloading %s ...", key, name)
        if kagglehub is None:
            _download_direct(handle, inner, dest)
        else:
            src = Path(kagglehub.dataset_download(handle, path=inner))
            if src.is_dir():  # pragma: no cover - defensive
                hits = sorted(src.rglob(Path(inner).name))
                if not hits:
                    raise FileNotFoundError(f"{inner} not found in downloaded archive at {src}")
                src = hits[0]
            shutil.copy2(src, dest)
        LOG.info("%-18s wrote %s (%.1f MB, sha256 %s)",
                 key, dest.name, dest.stat().st_size / 1e6, sha256_file(dest)[:16])
        written.append(dest)
    return written


def check(data_dir=None) -> dict:
    """Report which required files are on disk, with sizes and hashes."""
    out = {}
    for key, (handle, files) in SOURCES.items():
        d = target_dir(key, data_dir)
        rows = []
        for _inner, name in files:
            p = d / name
            rows.append({
                "file": name,
                "present": p.exists(),
                "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None,
                "sha256_16": sha256_file(p)[:16] if p.exists() else None,
            })
        out[key] = {"kaggle": handle, "dir": str(d), "files": rows,
                    "complete": all(r["present"] for r in rows)}
    return out


def main(argv=None) -> int:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("keys", nargs="*", help=f"dataset keys; default: {' '.join(DEFAULT_KEYS)}")
    ap.add_argument("--all", action="store_true", help="fetch every study dataset")
    ap.add_argument("--check", action="store_true", help="report what is on disk and exit")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)

    if args.check:
        for key, info in check(args.data_dir).items():
            mark = "OK  " if info["complete"] else "MISS"
            sizes = ", ".join(f"{r['file']} {r['size_mb'] or '-'}MB" for r in info["files"])
            print(f"[{mark}] {key:18s} {sizes}")
        return 0

    keys = list(args.keys) or (list(DEFAULT_KEYS) if args.all else list(DEFAULT_KEYS))
    total = sum(APPROX_MB.get(k, 0) for k in keys)
    LOG.info("fetching %d dataset(s), roughly %d MB total", len(keys), total)
    for k in keys:
        fetch(k, args.data_dir, args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
