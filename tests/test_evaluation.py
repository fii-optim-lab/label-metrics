import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from label_metrics.evaluation import evaluate_folders
from utils import NumpyReader, save_case


class TestEvaluateFolders(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.prediction_folder = self.root / "predictions"
        self.target_folder = self.root / "targets"

        self.prediction_folder.mkdir()
        self.target_folder.mkdir()

        self.labels = {
            "background": 0,
            "foreground": 1,
        }
        self.reader = NumpyReader()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def evaluate(self, **kwargs):
        arguments = {
            "prediction_folder": self.prediction_folder,
            "ground_truth_folder": self.target_folder,
            "labels": self.labels,
            "prediction_prefix": "pred_",
            "prediction_suffix": ".npy",
            "ground_truth_prefix": "gt_",
            "ground_truth_suffix": ".npy",
            "reader": self.reader,
            "workers": 1,
        }
        arguments.update(kwargs)

        return evaluate_folders(**arguments)

    def test_evaluate_single_case(self):
        prediction = np.array(
            [
                [0, 1],
                [0, 1],
            ],
            dtype=np.int64,
        )
        target = np.array(
            [
                [0, 0],
                [1, 1],
            ],
            dtype=np.int64,
        )

        save_case(
            self.prediction_folder,
            "pred_case_a.npy",
            prediction,
        )
        save_case(
            self.target_folder,
            "gt_case_a.npy",
            target,
        )

        result = self.evaluate()

        self.assertEqual(result.labels, self.labels)
        self.assertEqual(len(result.cases), 1)

        case = result.cases[0]

        self.assertEqual(case.case_name, "case_a")
        self.assertEqual(case.prediction_shape, (2, 2))
        self.assertEqual(case.target_shape, (2, 2))
        self.assertTrue(
            torch.equal(
                case.confusion,
                torch.tensor(
                    [
                        [1, 1],
                        [1, 1],
                    ]
                ),
            )
        )

    def test_evaluate_multiple_cases(self):
        save_case(
            self.prediction_folder,
            "pred_case_b.npy",
            np.array([0, 1], dtype=np.int64),
        )
        save_case(
            self.target_folder,
            "gt_case_b.npy",
            np.array([0, 1], dtype=np.int64),
        )
        save_case(
            self.prediction_folder,
            "pred_case_a.npy",
            np.array([1, 0], dtype=np.int64),
        )
        save_case(
            self.target_folder,
            "gt_case_a.npy",
            np.array([1, 0], dtype=np.int64),
        )

        result = self.evaluate()

        self.assertEqual(
            [case.case_name for case in result.cases],
            ["case_a", "case_b"],
        )
        self.assertTrue(
            torch.equal(
                result.global_confusion,
                torch.tensor(
                    [
                        [2, 0],
                        [0, 2],
                    ]
                ),
            )
        )

    def test_finds_files_recursively(self):
        save_case(
            self.prediction_folder / "nested",
            "pred_case.npy",
            np.array([0, 1], dtype=np.int64),
        )
        save_case(
            self.target_folder / "another_nested",
            "gt_case.npy",
            np.array([0, 1], dtype=np.int64),
        )

        result = self.evaluate()

        self.assertEqual(result.cases[0].case_name, "case")

    def test_ignores_nonmatching_prediction_files(self):
        save_case(
            self.prediction_folder,
            "ignored.npy",
            np.array([0, 1], dtype=np.int64),
        )
        save_case(
            self.prediction_folder,
            "pred_case.npy",
            np.array([0, 1], dtype=np.int64),
        )
        save_case(
            self.target_folder,
            "gt_case.npy",
            np.array([0, 1], dtype=np.int64),
        )

        result = self.evaluate()

        self.assertEqual(len(result.cases), 1)
        self.assertEqual(result.cases[0].case_name, "case")

    def test_default_ground_truth_suffix(self):
        save_case(
            self.prediction_folder,
            "pred_case.npy",
            np.array([0, 1], dtype=np.int64),
        )
        save_case(
            self.target_folder,
            "gt_case.npy",
            np.array([0, 1], dtype=np.int64),
        )

        result = self.evaluate(
            ground_truth_suffix=None,
        )

        self.assertEqual(result.cases[0].case_name, "case")

    def test_no_prediction_files(self):
        with self.assertRaisesRegex(
            FileNotFoundError,
            "No prediction files",
        ):
            self.evaluate()

    def test_missing_ground_truth(self):
        save_case(
            self.prediction_folder,
            "pred_case.npy",
            np.array([0, 1], dtype=np.int64),
        )

        with self.assertRaisesRegex(
            FileNotFoundError,
            "Missing ground-truth files",
        ):
            self.evaluate()

    def test_duplicate_prediction_case_name(self):
        save_case(
            self.prediction_folder / "folder_a",
            "pred_case.npy",
            np.array([0, 1], dtype=np.int64),
        )
        save_case(
            self.prediction_folder / "folder_b",
            "pred_case.npy",
            np.array([0, 1], dtype=np.int64),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate prediction case name",
        ):
            self.evaluate()

    def test_duplicate_ground_truth_case_name(self):
        save_case(
            self.prediction_folder,
            "pred_case.npy",
            np.array([0, 1], dtype=np.int64),
        )
        save_case(
            self.target_folder / "folder_a",
            "gt_case.npy",
            np.array([0, 1], dtype=np.int64),
        )
        save_case(
            self.target_folder / "folder_b",
            "gt_case.npy",
            np.array([0, 1], dtype=np.int64),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate ground-truth case name",
        ):
            self.evaluate()

    def test_shape_mismatch(self):
        save_case(
            self.prediction_folder,
            "pred_case.npy",
            np.zeros((2, 2), dtype=np.int64),
        )
        save_case(
            self.target_folder,
            "gt_case.npy",
            np.zeros((3, 2), dtype=np.int64),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Shape mismatch",
        ):
            self.evaluate()

    def test_negative_label(self):
        save_case(
            self.prediction_folder,
            "pred_case.npy",
            np.array([0, -1], dtype=np.int64),
        )
        save_case(
            self.target_folder,
            "gt_case.npy",
            np.array([0, 1], dtype=np.int64),
        )

        with self.assertRaisesRegex(
            ValueError,
            "negative labels",
        ):
            self.evaluate()

    def test_label_out_of_range(self):
        save_case(
            self.prediction_folder,
            "pred_case.npy",
            np.array([0, 2], dtype=np.int64),
        )
        save_case(
            self.target_folder,
            "gt_case.npy",
            np.array([0, 1], dtype=np.int64),
        )

        with self.assertRaisesRegex(
            ValueError,
            "contains label 2",
        ):
            self.evaluate()

    def test_prediction_folder_does_not_exist(self):
        with self.assertRaisesRegex(
            FileNotFoundError,
            "Folder does not exist",
        ):
            self.evaluate(
                prediction_folder=self.root / "missing",
            )

    def test_prediction_path_is_not_folder(self):
        path = self.root / "file"
        path.write_text("content", encoding="utf-8")

        with self.assertRaisesRegex(
            NotADirectoryError,
            "not a folder",
        ):
            self.evaluate(
                prediction_folder=path,
            )

    def test_empty_prediction_suffix(self):
        with self.assertRaisesRegex(
            ValueError,
            "prediction_suffix cannot be empty",
        ):
            self.evaluate(prediction_suffix="")

    def test_empty_ground_truth_suffix(self):
        with self.assertRaisesRegex(
            ValueError,
            "ground_truth_suffix cannot be empty",
        ):
            self.evaluate(ground_truth_suffix="")

    def test_supplied_reader_rejects_unsupported_type(self):
        path = self.prediction_folder / "pred_case.nii.gz"
        path.write_bytes(b"prediction")

        target = self.target_folder / "gt_case.nii.gz"
        target.write_bytes(b"target")

        with self.assertRaisesRegex(
            ValueError,
            "does not support file type",
        ):
            self.evaluate(
                prediction_suffix=".nii.gz",
                ground_truth_suffix=".nii.gz",
            )

    def test_unsupported_file_type_without_reader(self):
        prediction = self.prediction_folder / "pred_case.unknown"
        target = self.target_folder / "gt_case.unknown"

        prediction.write_bytes(b"prediction")
        target.write_bytes(b"target")

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported file type",
        ):
            evaluate_folders(
                prediction_folder=self.prediction_folder,
                ground_truth_folder=self.target_folder,
                labels=self.labels,
                prediction_prefix="pred_",
                prediction_suffix=".unknown",
                ground_truth_prefix="gt_",
                ground_truth_suffix=".unknown",
                workers=1,
            )

    def test_process_map_receives_requested_options(self):
        save_case(
            self.prediction_folder,
            "pred_case.npy",
            np.array([0, 1], dtype=np.int64),
        )
        save_case(
            self.target_folder,
            "gt_case.npy",
            np.array([0, 1], dtype=np.int64),
        )

        with patch(
            "label_metrics.evaluation.process_map",
        ) as process_map:
            process_map.return_value = []

            with self.assertRaisesRegex(
                ValueError,
                "at least one case",
            ):
                self.evaluate(
                    workers=4,
                    chunksize=3,
                )

        process_map.assert_called_once()

        worker_function, tasks = process_map.call_args.args
        options = process_map.call_args.kwargs

        self.assertEqual(worker_function.__name__, "_evaluate_task")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(options["max_workers"], 4)
        self.assertEqual(options["chunksize"], 3)
        self.assertEqual(options["desc"], "Evaluating")


class TestLabelValidation(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.predictions = self.root / "predictions"
        self.targets = self.root / "targets"
        self.predictions.mkdir()
        self.targets.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def evaluate_with_labels(self, labels):
        return evaluate_folders(
            prediction_folder=self.predictions,
            ground_truth_folder=self.targets,
            labels=labels,
            prediction_suffix=".npy",
            reader=NumpyReader(),
            workers=1,
        )

    def test_labels_must_be_dictionary(self):
        with self.assertRaisesRegex(
            TypeError,
            "must be a dictionary",
        ):
            self.evaluate_with_labels(["background"])

    def test_labels_cannot_be_empty(self):
        with self.assertRaisesRegex(
            ValueError,
            "cannot be empty",
        ):
            self.evaluate_with_labels({})

    def test_label_names_must_be_strings(self):
        with self.assertRaisesRegex(
            TypeError,
            "label name must be a string",
        ):
            self.evaluate_with_labels({0: 0})

    def test_label_values_must_be_integers(self):
        with self.assertRaisesRegex(
            TypeError,
            "must map to an integer",
        ):
            self.evaluate_with_labels(
                {
                    "background": 0,
                    "foreground": 1.0,
                }
            )

    def test_boolean_is_not_valid_label_index(self):
        with self.assertRaisesRegex(
            TypeError,
            "must map to an integer",
        ):
            self.evaluate_with_labels(
                {
                    "background": 0,
                    "foreground": True,
                }
            )

    def test_label_values_must_be_contiguous(self):
        with self.assertRaisesRegex(
            ValueError,
            "must be exactly 0 through 1",
        ):
            self.evaluate_with_labels(
                {
                    "background": 0,
                    "foreground": 2,
                }
            )

    def test_label_values_must_start_at_zero(self):
        with self.assertRaisesRegex(
            ValueError,
            "must be exactly 0 through 1",
        ):
            self.evaluate_with_labels(
                {
                    "background": 1,
                    "foreground": 2,
                }
            )

    def test_duplicate_label_values_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "must be exactly 0 through 1",
        ):
            self.evaluate_with_labels(
                {
                    "background": 0,
                    "foreground": 0,
                }
            )


if __name__ == "__main__":
    unittest.main()
