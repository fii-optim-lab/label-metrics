from pathlib import Path

from label_metrics.readers.base import Reader
from label_metrics.readers.sitk_reader import LabelReader


def file_type(path: str | Path) -> str:
    name = Path(path).name.lower()

    if name.endswith(".nii.gz"):
        return ".nii.gz"

    return Path(name).suffix


def get_reader(
    path: str | Path,
    reorient: bool = False,
) -> Reader:
    suffix = file_type(path)

    if suffix in LabelReader.accepted_file_types:
        return LabelReader(reorient=reorient)

    raise ValueError(
        f"Unsupported file type {suffix!r} for file {str(path)!r}."
    )