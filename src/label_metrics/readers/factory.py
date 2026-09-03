from pathlib import Path

from label_metrics.readers.base import BaseReader
from label_metrics.readers.simple_itk import SimpleITKReader


def file_type(path: str) -> str:
    name = Path(path).name.lower()

    if name.endswith(".nii.gz"):
        return ".nii.gz"

    return Path(name).suffix


def get_reader(
    path: str | Path,
    *,
    reorient: bool = False,
) -> BaseReader:
    suffix = file_type(path)

    if suffix in SimpleITKReader.accepted_file_types:
        return SimpleITKReader(reorient=reorient)

    raise ValueError(f"Unsupported file type {suffix!r} for file {str(path)!r}.")
