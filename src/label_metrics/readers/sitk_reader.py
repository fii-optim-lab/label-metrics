from numpy import ndarray


class LabelReader:
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
                f"Please install SimpleITK to read the following file types: {self.accepted_file_types}."
            ) from error

    def read(self, path: str) -> ndarray:
        import SimpleITK as sitk

        image = sitk.ReadImage(path)

        if self.reorient:
            image = sitk.DICOMOrient(image, "LPS")

        return sitk.GetArrayFromImage(image)
