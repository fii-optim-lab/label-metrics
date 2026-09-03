import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from label_metrics.loading import load_case
from utils import NumpyReader, save_case


class TestLoadCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary_directory.name)
        self.reader = NumpyReader()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_load_case(self):
        prediction_array = np.array(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=np.int64,
        )
        target_array = np.array(
            [
                [0, 1],
                [2, 2],
            ],
            dtype=np.int64,
        )

        prediction_path = save_case(
            self.folder,
            "prediction.npy",
            prediction_array,
        )
        target_path = save_case(
            self.folder,
            "target.npy",
            target_array,
        )

        result = load_case(
            prediction_path=prediction_path,
            target_path=target_path,
            case_name="case",
            num_labels=3,
            prediction_reader=self.reader,
            target_reader=self.reader,
        )

        self.assertEqual(result.prediction_shape, (2, 2))
        self.assertEqual(result.target_shape, (2, 2))
        self.assertTrue(result.prediction.is_contiguous())
        self.assertTrue(result.target.is_contiguous())

        self.assertTrue(
            torch.equal(
                result.prediction,
                torch.tensor(
                    prediction_array,
                    dtype=result.prediction.dtype,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                result.target,
                torch.tensor(
                    target_array,
                    dtype=result.target.dtype,
                ),
            )
        )

    def test_uses_smallest_safe_dtype(self):
        prediction_path = save_case(
            self.folder,
            "prediction.npy",
            np.array([0, 1], dtype=np.int64),
        )
        target_path = save_case(
            self.folder,
            "target.npy",
            np.array([1, 0], dtype=np.int64),
        )

        result = load_case(
            prediction_path=prediction_path,
            target_path=target_path,
            case_name="case",
            num_labels=3,
            prediction_reader=self.reader,
            target_reader=self.reader,
        )

        self.assertEqual(result.prediction.dtype, torch.uint8)
        self.assertEqual(result.target.dtype, torch.uint8)

    def test_makes_noncontiguous_arrays_contiguous(self):
        prediction = np.arange(16, dtype=np.int64).reshape(4, 4) % 2
        target = prediction.copy()

        prediction_path = save_case(
            self.folder,
            "prediction.npy",
            prediction.T,
        )
        target_path = save_case(
            self.folder,
            "target.npy",
            target.T,
        )

        result = load_case(
            prediction_path=prediction_path,
            target_path=target_path,
            case_name="case",
            num_labels=2,
            prediction_reader=self.reader,
            target_reader=self.reader,
        )

        self.assertTrue(result.prediction.is_contiguous())
        self.assertTrue(result.target.is_contiguous())

    def test_shape_mismatch(self):
        prediction_path = save_case(
            self.folder,
            "prediction.npy",
            np.zeros((2, 2), dtype=np.int64),
        )
        target_path = save_case(
            self.folder,
            "target.npy",
            np.zeros((3, 2), dtype=np.int64),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Shape mismatch",
        ):
            load_case(
                prediction_path=prediction_path,
                target_path=target_path,
                case_name="case",
                num_labels=2,
                prediction_reader=self.reader,
                target_reader=self.reader,
            )

    def test_negative_prediction_label(self):
        prediction_path = save_case(
            self.folder,
            "prediction.npy",
            np.array([0, -1], dtype=np.int64),
        )
        target_path = save_case(
            self.folder,
            "target.npy",
            np.array([0, 1], dtype=np.int64),
        )

        with self.assertRaisesRegex(
            ValueError,
            "negative labels",
        ):
            load_case(
                prediction_path=prediction_path,
                target_path=target_path,
                case_name="case",
                num_labels=2,
                prediction_reader=self.reader,
                target_reader=self.reader,
            )

    def test_negative_target_label(self):
        prediction_path = save_case(
            self.folder,
            "prediction.npy",
            np.array([0, 1], dtype=np.int64),
        )
        target_path = save_case(
            self.folder,
            "target.npy",
            np.array([0, -1], dtype=np.int64),
        )

        with self.assertRaisesRegex(
            ValueError,
            "negative labels",
        ):
            load_case(
                prediction_path=prediction_path,
                target_path=target_path,
                case_name="case",
                num_labels=2,
                prediction_reader=self.reader,
                target_reader=self.reader,
            )

    def test_prediction_label_out_of_range(self):
        prediction_path = save_case(
            self.folder,
            "prediction.npy",
            np.array([0, 2], dtype=np.int64),
        )
        target_path = save_case(
            self.folder,
            "target.npy",
            np.array([0, 1], dtype=np.int64),
        )

        with self.assertRaisesRegex(
            ValueError,
            "contains label 2",
        ):
            load_case(
                prediction_path=prediction_path,
                target_path=target_path,
                case_name="case",
                num_labels=2,
                prediction_reader=self.reader,
                target_reader=self.reader,
            )

    def test_target_label_out_of_range(self):
        prediction_path = save_case(
            self.folder,
            "prediction.npy",
            np.array([0, 1], dtype=np.int64),
        )
        target_path = save_case(
            self.folder,
            "target.npy",
            np.array([0, 2], dtype=np.int64),
        )

        with self.assertRaisesRegex(
            ValueError,
            "contains label 2",
        ):
            load_case(
                prediction_path=prediction_path,
                target_path=target_path,
                case_name="case",
                num_labels=2,
                prediction_reader=self.reader,
                target_reader=self.reader,
            )

    def test_empty_array(self):
        prediction_path = save_case(
            self.folder,
            "prediction.npy",
            np.empty(0, dtype=np.int64),
        )
        target_path = save_case(
            self.folder,
            "target.npy",
            np.empty(0, dtype=np.int64),
        )

        with self.assertRaisesRegex(
            ValueError,
            "is empty",
        ):
            load_case(
                prediction_path=prediction_path,
                target_path=target_path,
                case_name="case",
                num_labels=2,
                prediction_reader=self.reader,
                target_reader=self.reader,
            )

    def test_non_integer_array(self):
        prediction_path = save_case(
            self.folder,
            "prediction.npy",
            np.array([0.0, 1.0], dtype=np.float32),
        )
        target_path = save_case(
            self.folder,
            "target.npy",
            np.array([0, 1], dtype=np.int64),
        )

        with self.assertRaisesRegex(
            TypeError,
            "integer dtype",
        ):
            load_case(
                prediction_path=prediction_path,
                target_path=target_path,
                case_name="case",
                num_labels=2,
                prediction_reader=self.reader,
                target_reader=self.reader,
            )


if __name__ == "__main__":
    unittest.main()
