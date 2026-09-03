import unittest
from unittest.mock import patch

from label_metrics.readers.base import BaseReader
from label_metrics.readers.factory import file_type, get_reader
from label_metrics.readers.simple_itk import SimpleITKReader


class TestFileType(unittest.TestCase):
    def test_nii_gz(self):
        self.assertEqual(
            file_type("/folder/case.NII.GZ"),
            ".nii.gz",
        )

    def test_single_suffix(self):
        self.assertEqual(
            file_type("/folder/case.NRRD"),
            ".nrrd",
        )

    def test_no_suffix(self):
        self.assertEqual(
            file_type("/folder/case"),
            "",
        )


class TestReaderFactory(unittest.TestCase):
    def test_simple_itk_reader_inherits_base_reader(self):
        self.assertTrue(issubclass(SimpleITKReader, BaseReader))

    def test_unsupported_file_type(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported file type",
        ):
            get_reader("case.npy")

    def test_known_file_type_selects_simple_itk_reader(self):
        sentinel = object()

        with patch(
            "label_metrics.readers.factory.SimpleITKReader",
            return_value=sentinel,
        ) as reader_class:
            reader_class.accepted_file_types = (
                ".nii",
                ".nii.gz",
                ".mha",
                ".mhd",
                ".nrrd",
            )

            result = get_reader(
                "case.nii.gz",
                reorient=True,
            )

        self.assertIs(result, sentinel)
        reader_class.assert_called_once_with(reorient=True)


if __name__ == "__main__":
    unittest.main()
