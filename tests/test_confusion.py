import unittest

import torch

from label_metrics.confusion import py_confusion_matrix


class ConfusionMatrixTests(unittest.TestCase):
    __unittest_skip__ = True
    __unittest_skip_why__ = "Base test class"
    confusion_matrix = None

    def test_binary(self):
        target = torch.tensor([0, 0, 1, 1])
        prediction = torch.tensor([0, 1, 0, 1])

        result = self.confusion_matrix(prediction, target, 2)

        expected = torch.tensor(
            [
                [1, 1],
                [1, 1],
            ]
        )

        self.assertTrue(torch.equal(result, expected))

    def test_multiclass(self):
        target = torch.tensor([0, 0, 1, 1, 2, 2])
        prediction = torch.tensor([0, 1, 1, 2, 2, 0])

        result = self.confusion_matrix(prediction, target, 3)

        expected = torch.tensor(
            [
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 1],
            ]
        )

        self.assertTrue(torch.equal(result, expected))

    def test_multidimensional(self):
        target = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
            ]
        )
        prediction = torch.tensor(
            [
                [0, 2, 2],
                [1, 1, 0],
            ]
        )

        result = self.confusion_matrix(prediction, target, 3)

        expected = torch.tensor(
            [
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 1],
            ]
        )

        self.assertTrue(torch.equal(result, expected))

    def test_missing_label(self):
        target = torch.tensor([0, 0, 2, 2])
        prediction = torch.tensor([0, 2, 0, 2])

        result = self.confusion_matrix(prediction, target, 3)

        expected = torch.tensor(
            [
                [1, 0, 1],
                [0, 0, 0],
                [1, 0, 1],
            ]
        )

        self.assertTrue(torch.equal(result, expected))

    def test_empty_input(self):
        target = torch.empty(0, dtype=torch.int64)
        prediction = torch.empty(0, dtype=torch.int64)

        result = self.confusion_matrix(prediction, target, 3)

        expected = torch.zeros((3, 3), dtype=torch.int64)

        self.assertTrue(torch.equal(result, expected))


class TestPyConfusionMatrix(ConfusionMatrixTests):
    __unittest_skip__ = False
    confusion_matrix = staticmethod(py_confusion_matrix)


if __name__ == "__main__":
    unittest.main()
