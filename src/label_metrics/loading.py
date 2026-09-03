from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import ndarray
from torch import Tensor, from_numpy

from label_metrics.readers.base import BaseReader


@dataclass(frozen=True)
class LoadedCase:
    prediction: Tensor
    target: Tensor
    prediction_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


def load_case(
    prediction_path: str | Path,
    target_path: str | Path,
    case_name: str,
    num_labels: int,
    prediction_reader: BaseReader,
    target_reader: BaseReader,
) -> LoadedCase:
    prediction = prediction_reader.read(prediction_path)
    target = target_reader.read(target_path)

    _validate_array(
        prediction,
        name="Prediction",
        case_name=case_name,
        num_labels=num_labels,
    )
    _validate_array(
        target,
        name="Ground truth",
        case_name=case_name,
        num_labels=num_labels,
    )

    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch for case {case_name!r}: "
            f"prediction shape is {prediction.shape}, "
            f"ground-truth shape is {target.shape}."
        )

    prediction_shape = tuple(prediction.shape)
    target_shape = tuple(target.shape)
    dtype = _smallest_safe_dtype(num_labels)

    prediction = np.ascontiguousarray(
        prediction,
        dtype=dtype,
    )
    target = np.ascontiguousarray(
        target,
        dtype=dtype,
    )

    return LoadedCase(
        prediction=from_numpy(prediction),
        target=from_numpy(target),
        prediction_shape=prediction_shape,
        target_shape=target_shape,
    )


def _validate_array(
    array: ndarray,
    *,
    name: str,
    case_name: str,
    num_labels: int,
) -> None:
    if not isinstance(array, ndarray):
        raise TypeError(
            f"{name} reader output for case {case_name!r} "
            f"must be a NumPy array, got {type(array).__name__}."
        )

    if array.size == 0:
        raise ValueError(f"{name} for case {case_name!r} is empty.")

    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"{name} for case {case_name!r} must have an "
            f"integer dtype, got {array.dtype}."
        )

    minimum = int(array.min())
    maximum = int(array.max())

    if minimum < 0:
        raise ValueError(
            f"{name} for case {case_name!r} contains "
            f"negative labels. Minimum value: {minimum}."
        )

    if maximum >= num_labels:
        raise ValueError(
            f"{name} for case {case_name!r} contains "
            f"label {maximum}, but num_labels is {num_labels}."
        )


def _smallest_safe_dtype(
    num_labels: int,
) -> np.dtype:
    maximum_encoded_value = num_labels * num_labels - 1

    if maximum_encoded_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)

    if maximum_encoded_value <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)

    if maximum_encoded_value <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)

    if maximum_encoded_value <= np.iinfo(np.int64).max:
        return np.dtype(np.int64)

    raise ValueError(
        f"num_labels={num_labels} is too large for int64 confusion-matrix encoding."
    )
