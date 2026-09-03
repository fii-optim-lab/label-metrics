from pathlib import Path

import numpy as np
from numpy import ndarray

from label_metrics.readers.base import BaseReader


class NumpyReader(BaseReader):
    accepted_file_types = (".npy",)

    def read(self, path: str | Path) -> ndarray:
        return np.load(path)


def save_case(
    folder: Path,
    filename: str,
    array: ndarray,
) -> Path:
    path = folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return path
