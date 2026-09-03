import unittest

import torch

from label_metrics.metrics import (
    METRIC_NAMES,
    aggregate_case_metrics,
    metrics_from_confusion,
)


class TestMetricsFromConfusion(unittest.TestCase):
    def setUp(self):
        self.metric_index = {name: index for index, name in enumerate(METRIC_NAMES)}

    def test_perfect_metrics(self):
        confusion = torch.tensor(
            [
                [4, 0],
                [0, 6],
            ]
        )

        metrics = metrics_from_confusion(confusion)

        self.assertEqual(metrics.shape, (len(METRIC_NAMES), 2))
        self.assertTrue(torch.allclose(metrics, torch.ones_like(metrics)))

    def test_binary_metrics(self):
        confusion = torch.tensor(
            [
                [8, 2],
                [1, 9],
            ]
        )

        metrics = metrics_from_confusion(confusion)

        dice = metrics[self.metric_index["dice"]]
        iou = metrics[self.metric_index["iou"]]
        precision = metrics[self.metric_index["precision"]]
        recall = metrics[self.metric_index["recall"]]

        self.assertAlmostEqual(dice[1].item(), 18 / 21)
        self.assertAlmostEqual(iou[1].item(), 9 / 12)
        self.assertAlmostEqual(precision[1].item(), 9 / 11)
        self.assertAlmostEqual(recall[1].item(), 9 / 10)

    def test_absent_label_produces_nan(self):
        confusion = torch.tensor(
            [
                [5, 0, 0],
                [0, 5, 0],
                [0, 0, 0],
            ]
        )

        metrics = metrics_from_confusion(confusion)
        label_metrics = metrics[:, 2]

        self.assertTrue(torch.isnan(label_metrics[self.metric_index["dice"]]))
        self.assertTrue(torch.isnan(label_metrics[self.metric_index["iou"]]))
        self.assertTrue(torch.isnan(label_metrics[self.metric_index["precision"]]))
        self.assertTrue(torch.isnan(label_metrics[self.metric_index["recall"]]))

        self.assertEqual(
            label_metrics[self.metric_index["specificity"]].item(),
            1.0,
        )
        self.assertEqual(
            label_metrics[self.metric_index["accuracy"]].item(),
            1.0,
        )

    def test_missed_label_has_zero_dice_iou_and_recall(self):
        confusion = torch.tensor(
            [
                [5, 0],
                [3, 0],
            ]
        )

        metrics = metrics_from_confusion(confusion)

        self.assertEqual(
            metrics[self.metric_index["dice"], 1].item(),
            0.0,
        )
        self.assertEqual(
            metrics[self.metric_index["iou"], 1].item(),
            0.0,
        )
        self.assertEqual(
            metrics[self.metric_index["recall"], 1].item(),
            0.0,
        )
        self.assertTrue(torch.isnan(metrics[self.metric_index["precision"], 1]))


class TestAggregateCaseMetrics(unittest.TestCase):
    def test_mean_std_and_valid_counts(self):
        case_metrics = torch.tensor(
            [
                [
                    [1.0, float("nan")],
                    [0.5, 0.2],
                ],
                [
                    [0.5, 0.8],
                    [1.0, float("nan")],
                ],
            ],
            dtype=torch.float64,
        )

        mean, std, valid = aggregate_case_metrics(case_metrics)

        expected_mean = torch.tensor(
            [
                [0.75, 0.8],
                [0.75, 0.2],
            ],
            dtype=torch.float64,
        )
        expected_std = torch.tensor(
            [
                [0.25, 0.0],
                [0.25, 0.0],
            ],
            dtype=torch.float64,
        )
        expected_valid = torch.tensor(
            [
                [2, 1],
                [2, 1],
            ]
        )

        self.assertTrue(torch.allclose(mean, expected_mean))
        self.assertTrue(torch.allclose(std, expected_std))
        self.assertTrue(torch.equal(valid, expected_valid))

    def test_all_nan_values_remain_nan(self):
        case_metrics = torch.full(
            (2, 1, 1),
            float("nan"),
            dtype=torch.float64,
        )

        mean, std, valid = aggregate_case_metrics(case_metrics)

        self.assertTrue(torch.isnan(mean).all())
        self.assertTrue(torch.isnan(std).all())
        self.assertTrue(torch.equal(valid, torch.zeros_like(valid)))

    def test_single_valid_case_has_zero_std(self):
        case_metrics = torch.tensor(
            [
                [[0.75]],
                [[float("nan")]],
            ],
            dtype=torch.float64,
        )

        mean, std, valid = aggregate_case_metrics(case_metrics)

        self.assertEqual(mean.item(), 0.75)
        self.assertEqual(std.item(), 0.0)
        self.assertEqual(valid.item(), 1)

    def test_invalid_shape(self):
        with self.assertRaisesRegex(
            ValueError,
            "num_cases, num_metrics, num_labels",
        ):
            aggregate_case_metrics(torch.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
