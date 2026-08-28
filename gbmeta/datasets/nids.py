"""The five external IDS benchmarks (plus Bot-IoT), each in its smallest usable form.

Every file choice below is the *small* variant, so the whole cross-dataset study
fits a free Colab T4 with room to spare:

===============  ==========================================  ========  =======  =========
key              file                                        rows      classes  CSV size
===============  ==========================================  ========  =======  =========
edge_iiotset     ML-EdgeIIoT-dataset.csv                      157,800       15    78 MiB
nslkdd           KDDTrain+.txt                                125,973       23    18 MiB
unsw_nb15        UNSW_NB15_testing-set.csv (see note)         175,341       10    31 MiB
ton_iot          Train_Test_Network.csv                       461,043       10    67 MiB
cicids2017       Wednesday-workingHours.pcap_ISCX.csv         692,703        6   215 MiB
botiot           ..._Final_10_best_Testing.csv                733,705        5    88 MiB
===============  ==========================================  ========  =======  =========

Each loader encodes the file-level traps that silently corrupt results if
missed. They are documented inline because "we used CICIDS2017" is not a
reproducible statement -- which of the two incompatible 79/85-column layouts,
which encoding, and whether the blank padding rows were dropped all change the
numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..utils import LOG
from .base import DatasetSpec, LoadedDataset, assemble, register, sanitise_columns

# --------------------------------------------------------------------------
# Streaming sampler for the files that do not fit comfortably in RAM
# --------------------------------------------------------------------------
def read_csv_sampled(
    path, label_col: str, max_rows: int, min_rows_per_class: int, seed: int,
    chunksize: int = 200_000, **read_kwargs,
) -> pd.DataFrame:
    """Two-pass, class-aware subsample of a CSV that is too large to hold whole.

    Pass 1 reads only the label column and counts classes. Pass 2 keeps each row
    with probability ``target_c / count_c``, so the sample is uniform *within*
    each class while the class sizes follow the same floor-plus-proportional
    allocation used everywhere else.

    Two cheap passes beat one clever one: reservoir sampling over DataFrames
    needs row-level bookkeeping that is slower than simply reading the file
    twice, and the Bernoulli scheme is trivially auditable.
    """
    path = Path(path)
    counts = pd.Series(dtype="int64")
    for chunk in pd.read_csv(path, usecols=[label_col], chunksize=chunksize, **read_kwargs):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        counts = counts.add(chunk[label_col].astype(str).str.strip().value_counts(), fill_value=0)
    counts = counts.astype(int)
    total = int(counts.sum())
    LOG.info("%s: %d rows, %d classes (pass 1)", path.name, total, len(counts))
    if total <= max_rows:
        return pd.read_csv(path, **read_kwargs)

    floor = np.minimum(counts.values, min_rows_per_class)
    budget = max_rows - int(floor.sum())
    leftover = counts.values - floor
    if budget <= 0:
        target = np.maximum(1, (floor * max_rows / max(floor.sum(), 1)).astype(int))
    else:
        share = leftover / max(leftover.sum(), 1)
        target = floor + np.minimum((share * budget).astype(int), leftover)
    keep_p = dict(zip(counts.index, np.clip(target / np.maximum(counts.values, 1), 0, 1)))

    rng = np.random.default_rng(seed)
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize, **read_kwargs):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        lab = chunk[label_col].astype(str).str.strip()
        p = lab.map(keep_p).fillna(1.0).to_numpy()
        parts.append(chunk[rng.random(len(chunk)) < p])
    out = pd.concat(parts, ignore_index=True)
    LOG.info("%s: streamed subsample %d -> %d rows (pass 2)", path.name, total, len(out))
    return out


def _resolve(spec: DatasetSpec, filename: str, data_dir=None) -> Path:
    """Locate a dataset file, searching the kagglehub cache layout too."""
    root = Path(data_dir or DATA_DIR) / spec.key
    direct = root / Path(filename).name
    if direct.exists():
        return direct
    nested = root / filename
    if nested.exists():
        return nested
    hits = sorted(root.rglob(Path(filename).name)) if root.exists() else []
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"{spec.key}: could not find {filename!r} under {root}.\n"
        f"  Fetch it with:  python -m gbmeta.datasets.fetch {spec.key}\n"
        f"  Or download manually from: {spec.source}"
    )


# ==========================================================================
# Edge-IIoTset
# ==========================================================================
#: The 15 columns the dataset author's own preprocessing notebook drops. They
#: are raw payloads, hostnames, IPs, ports and the (corrupt) capture timestamp:
#: exactly the columns that let a model identify the attack without inspecting
#: any behaviour.
EDGE_DROP = (
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
)
EDGE_CAT = (
    "http.request.method", "http.referer", "http.request.version",
    "dns.qry.name.len", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic",
)

EDGE = DatasetSpec(
    key="edge_iiotset",
    display="Edge-IIoTset",
    label_col="Attack_type",
    binary_label_col="Attack_label",
    drop_cols=EDGE_DROP,
    timestamp_col=None,  # frame.time is corrupt: see note in load_edge_iiotset
    categorical_cols=EDGE_CAT,
    source="Kaggle mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
    kaggle="mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
    files=("Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv",),
    license="Academic use granted in perpetuity; commercial use by permission (Ferrag et al., 2022)",
    notes=(
        "157,800 rows x 63 cols, 15 classes. The 2.2M-row DNN-EdgeIIoT-dataset.csv "
        "variant is selectable with variant='dnn' but is a 1.2 GB download."
    ),
)

EDGE_VARIANTS = {
    "ml": "Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv",
    "dnn": "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv",
}


def load_edge_iiotset(
    max_rows=200_000, min_rows_per_class=200, min_class_support=30, seed=42,
    dedup="global", variant="ml", data_dir=None, **_,
) -> LoadedDataset:
    """Edge-IIoTset, the paper's original benchmark.

    ``frame.time`` is deliberately *not* used as a timestamp. The published CSV
    was written with an unquoted Wireshark time string, so the comma inside
    ``"Dec  2, 2021 11:44:10 EET"`` split the field and the month/day were lost;
    ~10% of rows contain column-shifted garbage in that position. Only the year
    and time-of-day survive, which cannot order a multi-day capture. Any
    "temporal" split on this column would be fiction, so drift experiments use
    ToN-IoT instead.
    """
    path = _resolve(EDGE, EDGE_VARIANTS[variant], data_dir)
    if variant == "dnn":
        df = read_csv_sampled(path, EDGE.label_col, max_rows * 3, min_rows_per_class * 3,
                              seed, low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    return assemble(
        EDGE, [df], max_rows, min_rows_per_class, min_class_support, seed, dedup,
        extra_provenance={"source_file": str(path.name), "variant": variant,
                          "timestamp_usable": False,
                          "timestamp_note": "frame.time corrupted by unquoted comma in source CSV"},
    )


# ==========================================================================
# NSL-KDD
# ==========================================================================
NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_type", "difficulty",
]

NSLKDD = DatasetSpec(
    key="nslkdd",
    display="NSL-KDD",
    label_col="attack_type",
    binary_label_col=None,
    drop_cols=("difficulty",),
    timestamp_col=None,
    categorical_cols=("protocol_type", "service", "flag"),
    source="Kaggle hassan06/nslkdd (the official UNB page no longer serves the files)",
    kaggle="hassan06/nslkdd",
    files=("KDDTrain+.txt",),
    license="Redistribution permitted with citation to Tavallaee et al., CISDA 2009",
    notes="125,973 rows x 43 fields, headerless, 23 classes. No timestamp of any kind.",
)


def load_nslkdd(
    max_rows=200_000, min_rows_per_class=200, min_class_support=30, seed=42,
    dedup="global", data_dir=None, **_,
) -> LoadedDataset:
    """NSL-KDD.

    Two traps handled here. The file has **no header row**, so column names come
    from the published schema. And field 43 (``difficulty``) is the number of 21
    reference classifiers that got the record right -- a direct function of the
    label. Leaving it in inflates accuracy by several points for free, and it is
    dropped by the spec.
    """
    path = _resolve(NSLKDD, "KDDTrain+.txt", data_dir)
    df = pd.read_csv(path, header=None, names=NSLKDD_COLUMNS, low_memory=False)
    return assemble(
        NSLKDD, [df], max_rows, min_rows_per_class, min_class_support, seed, dedup,
        extra_provenance={"source_file": path.name,
                          "leaky_column_dropped": "difficulty (count of reference "
                                                  "classifiers that got the row right)"},
    )


# ==========================================================================
# UNSW-NB15
# ==========================================================================
UNSW = DatasetSpec(
    key="unsw_nb15",
    display="UNSW-NB15",
    label_col="attack_cat",
    binary_label_col="label",
    drop_cols=("id",),
    timestamp_col=None,  # partition files carry no time column
    categorical_cols=("proto", "service", "state"),
    source="Kaggle mrwellsdavid/unsw-nb15",
    kaggle="mrwellsdavid/unsw-nb15",
    files=("UNSW_NB15_testing-set.csv",),
    license="Academic use granted in perpetuity (Moustafa & Slay, 2015)",
    notes=(
        "The file NAMED testing-set holds 175,341 rows and the one named "
        "training-set holds 82,332 -- this Kaggle mirror has them swapped "
        "relative to UNSW's documentation. The larger partition is used."
    ),
)


def load_unsw_nb15(
    max_rows=200_000, min_rows_per_class=200, min_class_support=30, seed=42,
    dedup="global", data_dir=None, **_,
) -> LoadedDataset:
    """UNSW-NB15 partition file.

    ``encoding="utf-8-sig"`` is not optional: the file starts with a UTF-8 BOM,
    so without it the first column is named ``'\\ufeffid'`` and the ``id`` drop
    silently does nothing -- leaving a monotonic row index in the feature matrix.
    """
    path = _resolve(UNSW, "UNSW_NB15_testing-set.csv", data_dir)
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    return assemble(
        UNSW, [df], max_rows, min_rows_per_class, min_class_support, seed, dedup,
        extra_provenance={"source_file": path.name, "rows_in_file": int(len(df)),
                          "note": "Kaggle mirror swaps the train/test file names"},
    )


# ==========================================================================
# ToN-IoT  (the drift dataset)
# ==========================================================================
TONIOT = DatasetSpec(
    key="ton_iot",
    display="ToN-IoT",
    label_col="type",
    binary_label_col="label",
    drop_cols=(
        "src_ip", "src_port", "dst_ip", "dst_port",
        # Free-text fields that name the attack outright.
        "dns_query", "ssl_subject", "ssl_issuer", "http_uri", "http_user_agent",
        "weird_addl", "http_referrer",
    ),
    timestamp_col="ts",
    categorical_cols=(
        "proto", "service", "conn_state", "dns_qclass", "dns_qtype", "dns_rcode",
        "dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_version", "ssl_cipher",
        "ssl_resumed", "ssl_established", "http_method", "http_version",
        "http_orig_mime_types", "http_resp_mime_types", "weird_name", "weird_notice",
    ),
    source="Kaggle alaaelmor/ton-iot-train-test-network",
    kaggle="alaaelmor/ton-iot-train-test-network",
    files=("Train_Test_Network.csv",),
    license="Academic use granted in perpetuity (Moustafa, 2021)",
    notes=(
        "461,043 rows x 45 cols, 10 classes. The only one of the six with a "
        "clean Unix-epoch timestamp, so it carries the temporal-drift study."
    ),
)


def load_ton_iot(
    max_rows=200_000, min_rows_per_class=200, min_class_support=30, seed=42,
    dedup="global", data_dir=None, **_,
) -> LoadedDataset:
    """ToN-IoT.

    Missing values are the literal string ``'-'``, not empty cells; without
    ``na_values`` every numeric column is inferred as ``object`` and silently
    one-hot encoded into thousands of useless columns.
    """
    path = _resolve(TONIOT, "Train_Test_Network.csv", data_dir)
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig", na_values=["-"])
    df = sanitise_columns(df)
    if "ts" in df.columns:
        # Epoch seconds -> datetime, kept as a separate column so `assemble`
        # can build the drift ordering before `ts` is dropped as an identifier.
        df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce"), unit="s")
    return assemble(
        TONIOT, [df], max_rows, min_rows_per_class, min_class_support, seed, dedup,
        extra_provenance={"source_file": path.name, "timestamp_usable": True},
    )


# ==========================================================================
# CICIDS2017
# ==========================================================================
CICIDS_FILES = {
    "wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
    "tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "thursday_web": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "friday_portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "friday_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "friday_bot": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
}
#: Day-files combined by default. Wednesday alone has 6 classes; adding the
#: three afternoon files brings the total to 11 attack types without pulling in
#: Monday, which is 100% benign and would only dilute the sample.
CICIDS_DEFAULT = ("wednesday", "thursday_web", "friday_portscan", "friday_ddos", "friday_bot")

CICIDS = DatasetSpec(
    key="cicids2017",
    display="CICIDS2017",
    label_col="Label",
    binary_label_col=None,
    drop_cols=("Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp",
               "Fwd Header Length.1"),
    timestamp_col=None,  # MachineLearningCVE variant has none; see notes
    categorical_cols=(),
    source="Kaggle chethuhn/network-intrusion-dataset (MachineLearningCVE, 79 columns)",
    kaggle="chethuhn/network-intrusion-dataset",
    files=tuple(CICIDS_FILES[k] for k in CICIDS_DEFAULT),
    license="Citation to Sharafaldin et al., ICISSP 2018 is mandatory",
    notes=(
        "The 79-column MachineLearningCVE layout. The 85-column TrafficLabelling "
        "layout has the same filenames but different columns and a broken "
        "12-hour timestamp; it is not used here."
    ),
)


def load_cicids2017(
    max_rows=200_000, min_rows_per_class=200, min_class_support=30, seed=42,
    dedup="global", days=CICIDS_DEFAULT, data_dir=None, **_,
) -> LoadedDataset:
    """CICIDS2017, combined day-files.

    Four file-level traps, all handled here:

    1. ``encoding="latin-1"``. The web-attack labels contain cp1252 byte 0x96
       (an en-dash), which makes a default UTF-8 read raise outright.
    2. Column names carry inconsistent leading spaces; they are stripped, and
       the label becomes ``Label`` rather than ``' Label'``.
    3. ``' Fwd Header Length'`` appears **twice**; pandas renames the second to
       ``Fwd Header Length.1`` and the duplicate is dropped by the spec.
    4. Flow-rate columns contain literal ``inf``; those become NaN and are
       imputed inside the fold-fitted pipeline, not globally.
    """
    per_file = max(1, max_rows // max(len(days), 1)) * 3
    frames = []
    for d in days:
        path = _resolve(CICIDS, CICIDS_FILES[d], data_dir)
        df = read_csv_sampled(
            path, "Label", per_file, min_rows_per_class * 2, seed,
            encoding="latin-1", low_memory=False,
        )
        df = sanitise_columns(df)
        # The TrafficLabelling variant pads one file with ~288k blank rows.
        df = df.dropna(how="all")
        # Normalise the cp1252 en-dash so class names are ASCII in every table.
        df["Label"] = df["Label"].astype(str).str.replace("\x96", "-", regex=False).str.strip()
        frames.append(df)
        LOG.info("cicids2017/%s: %d rows kept", d, len(df))

    return assemble(
        CICIDS, frames, max_rows, min_rows_per_class, min_class_support, seed, dedup,
        extra_provenance={"day_files": list(days), "layout": "MachineLearningCVE (79 columns)"},
    )


# ==========================================================================
# Bot-IoT (optional -- included for completeness, flagged as pathological)
# ==========================================================================
BOTIOT = DatasetSpec(
    key="botiot",
    display="Bot-IoT",
    label_col="category",
    binary_label_col="attack",
    drop_cols=("pkSeqID", "saddr", "sport", "daddr", "dport", "subcategory"),
    timestamp_col=None,
    categorical_cols=("proto", "state_number"),
    source="Kaggle liuwoo/botiot-2018",
    kaggle="liuwoo/botiot-2018",
    files=("UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv",),
    license="Academic use granted in perpetuity (Koroniotis et al., 2019)",
    notes=(
        "733,705 rows, 5 classes, but Normal has 107 rows (0.015%) and Theft has "
        "14. Macro-F1 here is dominated by two double-digit-support classes; the "
        "dataset is reported with that caveat rather than excluded."
    ),
)


def load_botiot(
    max_rows=200_000, min_rows_per_class=200, min_class_support=10, seed=42,
    dedup="global", data_dir=None, **_,
) -> LoadedDataset:
    """Bot-IoT '10 best features' testing partition.

    ``pkSeqID`` is a global monotonic capture-order identifier spanning the whole
    3.6M-row capture, so it encodes which attack scenario a row came from; it is
    dropped, not merely deprioritised. ``sport``/``dport`` contain hexadecimal
    strings such as ``'0x0303'`` and are dropped rather than coerced.
    """
    path = _resolve(BOTIOT, "UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv", data_dir)
    df = pd.read_csv(path, low_memory=False)
    return assemble(
        BOTIOT, [df], max_rows, min_rows_per_class, min_class_support, seed, dedup,
        extra_provenance={
            "source_file": path.name,
            "caveat": "Normal=107 rows, Theft=14 rows; per-class metrics for those "
                      "two classes are not statistically meaningful",
        },
    )


# --------------------------------------------------------------------------
for _spec, _fn in (
    (EDGE, load_edge_iiotset),
    (NSLKDD, load_nslkdd),
    (UNSW, load_unsw_nb15),
    (TONIOT, load_ton_iot),
    (CICIDS, load_cicids2017),
    (BOTIOT, load_botiot),
):
    register(_spec, _fn)

#: The five datasets used for the headline cross-dataset comparison. Bot-IoT is
#: available but excluded by default because two of its five classes have
#: double-digit support.
STUDY_DATASETS = ("edge_iiotset", "nslkdd", "unsw_nb15", "ton_iot", "cicids2017")
