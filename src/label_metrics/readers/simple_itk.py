from pathlib import Path

from numpy import ndarray

from label_metrics.readers.base import BaseReader


class SimpleITKReader(BaseReader):
    accepted_file_types = (
        ".nii",
        ".nii.gz",
        ".mha",
        ".mhd",
        ".nrrd",
    )

    def __init__(self, reorient: bool = False) -> None:
        self.reorient = reorient

        try:
            import SimpleITK
        except ImportError as error:
            raise RuntimeError(
                "SimpleITK is required to read "
                f"{self.accepted_file_types}. "
                "Install it with `pip install SimpleITK`."
            ) from error

    def read(self, path: str | Path) -> ndarray:
        import SimpleITK as sitk

        image = sitk.ReadImage(str(path))

        if self.reorient:
            image = sitk.DICOMOrient(image, "LPS")

        return sitk.GetArrayFromImage(image)
