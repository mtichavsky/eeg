"""Tests that dataset objects survive pickling.

Python 3.14 changed the default multiprocessing start method on Linux from ``fork`` to
``forkserver``. Under ``forkserver`` the ``DataLoader`` pickles the dataset to hand it to
each worker process, which previously never happened. The dataset classes cache their
preprocessing functions in ``functools.lru_cache`` wrappers stored as instance attributes,
and those wrappers are not picklable, so training with ``num_workers > 0`` crashed with::

    AttributeError: 'functools._lru_cache_wrapper' object has no attribute '__qualname__'
"""

import pickle

import pytest

from thesis.dataset import CANEDataset, IDUNDataset, MDDDataset, SADDataset, SpectrogramDataset


@pytest.mark.parametrize(
    ("cls", "kwargs", "cache_attr"),
    [
        (MDDDataset, {"channel": "all"}, "_load_and_preprocess_edf_raw_file_cached"),
        (SADDataset, {"channel": "all"}, "_load_and_preprocess_edf_raw_file_cached"),
        (CANEDataset, {"channel": "all"}, "_load_and_preprocess_cane_raw_file"),
        (IDUNDataset, {}, "_load_cached"),
    ],
)
def test_edf_datasets_roundtrip_pickle(tmp_path, cls, kwargs, cache_attr):
    """Each raw dataset pickles and unpickles with a working cache wrapper."""
    dataset = cls(data_dir=tmp_path, cache_size=8, **kwargs)
    assert hasattr(dataset, cache_attr)

    restored = pickle.loads(pickle.dumps(dataset))

    assert callable(getattr(restored, cache_attr))
    assert restored.files == dataset.files


def test_spectrogram_dataset_roundtrip_pickle(tmp_path):
    """SpectrogramDataset (and its wrapped dataset) pickle end to end."""
    inner = MDDDataset(data_dir=tmp_path, cache_size=8, channel="all")
    spec = SpectrogramDataset(inner, source_fs=MDDDataset.FS, cache_size=8)
    assert hasattr(spec, "_get_item_cached")

    restored = pickle.loads(pickle.dumps(spec))

    assert callable(restored._get_item_cached)
    assert isinstance(restored.dataset, MDDDataset)
    assert callable(restored.dataset._load_and_preprocess_edf_raw_file_cached)


def test_pickle_without_cache(tmp_path):
    """Datasets built with cache_size=None (plain function attr) also pickle."""
    dataset = MDDDataset(data_dir=tmp_path, cache_size=None, channel="all")
    restored = pickle.loads(pickle.dumps(dataset))
    assert callable(restored._load_and_preprocess_edf_raw_file_cached)
