"""Dataset registry. Importing this module registers every available loader."""
from .base import (  # noqa: F401
    DatasetSpec, LoadedDataset, get_spec, list_datasets, load_dataset, register,
)
from . import synthetic  # noqa: F401,E402
from . import nids  # noqa: F401,E402
from .nids import STUDY_DATASETS  # noqa: F401,E402

__all__ = [
    "DatasetSpec", "LoadedDataset", "get_spec", "list_datasets", "load_dataset",
    "register", "STUDY_DATASETS",
]
