import json
import tempfile
import unittest
from pathlib import Path

import torch

from label_metrics.metrics import METRIC_NAMES
from label_metrics.result import CaseResult, EvaluationResult


class TestEvaluationResult(unittest.TestCase):
    def setUp(self):
        self.case_a = CaseResult(
            case_name="case_a",
            prediction_shape=(2, 2),
            target_shape=(2, 2),
            confusion=torch.tensor(
                [
                    [2, 0],
                    [1, 1],
                ]
            ),
        )
        self.case_b = CaseResult(
            case_name="case_b",
            prediction_shape=(2, 2),
            target_shape=(2, 2),
            confusion=torch.tensor(
                [
                    [1, 1],
                    [0, 2],
                ]
            ),
        )
        self.result = EvaluationResult(
            labels={
                "background": 0,
                "foreground": 1,
            },
            cases=(self.case_a, self.case_b),
        )

    def test_voxel_count_is_derived_from_confusion(self):
        self.assertEqual(self.case_a.voxel_count, 4)
        self.assertEqual(self.case_b.voxel_count, 4)
        self.assertEqual(self.result.total_voxel_count, 8)

    def test_global_confusion(self):
        expected = torch.tensor(
            [
                [3, 1],
                [1, 3],
            ]
        )

        self.assertTrue(
            torch.equal(
                self.result.global_confusion,
                expected,
            )
        )

    def test_global_metrics(self):
        metrics = self.result.global_metrics()

        dice_index = METRIC_NAMES.index("dice")
        iou_index = METRIC_NAMES.index("iou")

        self.assertAlmostEqual(
            metrics[dice_index, 0].item(),
            0.75,
        )
        self.assertAlmostEqual(
            metrics[dice_index, 1].item(),
            0.75,
        )
        self.assertAlmostEqual(
            metrics[iou_index, 0].item(),
            0.6,
        )
        self.assertAlmostEqual(
            metrics[iou_index, 1].item(),
            0.6,
        )

    def test_case_metrics_shape(self):
        metrics = self.result.case_metrics()

        self.assertEqual(
            metrics.shape,
            (
                2,
                len(METRIC_NAMES),
                2,
            ),
        )

    def test_case_metric_summary_shape(self):
        mean, std, valid = self.result.case_metric_summary()

        expected_shape = (len(METRIC_NAMES), 2)

        self.assertEqual(mean.shape, expected_shape)
        self.assertEqual(std.shape, expected_shape)
        self.assertEqual(valid.shape, expected_shape)

    def test_empty_result_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "at least one case",
        ):
            EvaluationResult(
                labels={"background": 0},
                cases=(),
            )

    def test_to_dict_does_not_store_voxel_count(self):
        data = self.result.to_dict()

        self.assertNotIn(
            "voxel_count",
            data["cases"][0],
        )

    def test_serialize_and_deserialize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"

            self.result.serialize(path)
            loaded = EvaluationResult.deserialize(path)

        self.assertEqual(loaded.labels, self.result.labels)
        self.assertEqual(len(loaded.cases), 2)

        for actual, expected in zip(
            loaded.cases,
            self.result.cases,
        ):
            self.assertEqual(
                actual.case_name,
                expected.case_name,
            )
            self.assertEqual(
                actual.prediction_shape,
                expected.prediction_shape,
            )
            self.assertEqual(
                actual.target_shape,
                expected.target_shape,
            )
            self.assertTrue(
                torch.equal(
                    actual.confusion,
                    expected.confusion,
                )
            )

    def test_metric_report_structure(self):
        report = self.result.metric_report()

        self.assertIn("global_confusion", report)
        self.assertIn("global_metrics", report)
        self.assertIn("case_summary", report)
        self.assertIn("cases", report)

        self.assertIn(
            "background",
            report["global_metrics"],
        )
        self.assertIn(
            "foreground",
            report["global_metrics"],
        )

        foreground = report["global_metrics"]["foreground"]

        for metric_name in METRIC_NAMES:
            self.assertIn(metric_name, foreground)

    def test_metrics_json_is_strict_json(self):
        absent_label_result = EvaluationResult(
            labels={
                "background": 0,
                "absent": 1,
            },
            cases=(
                CaseResult(
                    case_name="case",
                    prediction_shape=(2,),
                    target_shape=(2,),
                    confusion=torch.tensor(
                        [
                            [2, 0],
                            [0, 0],
                        ]
                    ),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            absent_label_result.save_metrics_json(path)

            content = path.read_text(encoding="utf-8")
            data = json.loads(content)

        self.assertNotIn("NaN", content)
        self.assertIsNone(data["global_metrics"]["absent"]["dice"])
        self.assertIsNone(data["global_metrics"]["absent"]["iou"])

    def test_text_report_contains_confusion_and_metrics(self):
        report = self.result.text_report()

        self.assertIn("Global confusion matrix", report)
        self.assertIn("Global metrics", report)
        self.assertIn("Per-case metric summary", report)
        self.assertIn("Per-case metrics", report)
        self.assertIn("background [0]", report)
        self.assertIn("foreground [1]", report)
        self.assertIn("dice:", report)
        self.assertIn("iou:", report)
        self.assertIn("precision:", report)
        self.assertIn("recall:", report)

    def test_text_report_section_order(self):
        report = self.result.text_report()

        global_confusion = report.index("Global confusion matrix")
        global_metrics = report.index("Global metrics")
        summary = report.index("Per-case metric summary")
        cases = report.index("Per-case metrics")

        self.assertLess(global_confusion, global_metrics)
        self.assertLess(global_metrics, summary)
        self.assertLess(summary, cases)

    def test_save_text_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.txt"
            self.result.save_text_report(path)

            content = path.read_text(encoding="utf-8")

        self.assertEqual(content, self.result.text_report())


if __name__ == "__main__":
    unittest.main()
