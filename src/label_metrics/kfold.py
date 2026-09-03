import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from label_metrics.metrics import (
    METRIC_NAMES,
    aggregate_case_metrics,
    metrics_from_confusion,
)
from label_metrics.result import EvaluationResult


@dataclass(frozen=True)
class KFoldResult:
    folds: dict[str, EvaluationResult]

    def __post_init__(self) -> None:
        if not self.folds:
            raise ValueError("KFoldResult must contain at least one fold.")

        labels = None

        for fold_name, result in self.folds.items():
            if labels is None:
                labels = result.labels
            elif result.labels != labels:
                raise ValueError(
                    f"Fold {fold_name!r} has a different label dictionary."
                )

    @property
    def labels(self) -> dict[str, int]:
        return next(iter(self.folds.values())).labels

    @property
    def num_folds(self) -> int:
        return len(self.folds)

    @property
    def global_confusion(self) -> Tensor:
        results = iter(self.folds.values())
        confusion = next(results).global_confusion.clone()

        for result in results:
            confusion += result.global_confusion

        return confusion

    def global_metrics(self) -> Tensor:
        return metrics_from_confusion(self.global_confusion)

    def fold_metrics(self) -> Tensor:
        return torch.stack([result.global_metrics() for result in self.folds.values()])

    def fold_metric_summary(
        self,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return aggregate_case_metrics(self.fold_metrics())

    @classmethod
    def load(
        cls,
        paths: dict[str, str | Path],
    ) -> KFoldResult:
        return cls(
            folds={
                fold_name: EvaluationResult.deserialize(path)
                for fold_name, path in paths.items()
            }
        )

    @classmethod
    def load_folder(
        cls,
        folder: str | Path,
        *,
        pattern: str = "fold_*/validation/evaluation.json",
    ) -> KFoldResult:
        folder = Path(folder)
        paths = sorted(folder.glob(pattern))

        if not paths:
            raise FileNotFoundError(
                f"No fold results matching {pattern!r} were found in {str(folder)!r}."
            )

        folds: dict[str, EvaluationResult] = {}

        for path in paths:
            fold_name = path.parent.parent.name

            if fold_name in folds:
                raise ValueError(f"Duplicate fold name {fold_name!r}.")

            folds[fold_name] = EvaluationResult.deserialize(path)

        return cls(folds=folds)

    def metric_report(self) -> dict[str, Any]:
        global_metrics = self.global_metrics()
        fold_metrics = self.fold_metrics()
        mean, std, valid = aggregate_case_metrics(fold_metrics)

        label_names = self._label_names()

        return {
            "labels": dict(self.labels),
            "num_folds": self.num_folds,
            "global_confusion": self.global_confusion.tolist(),
            "global_metrics": self._metrics_by_label(
                global_metrics,
                label_names,
            ),
            "fold_summary": self._summary_by_label(
                mean,
                std,
                valid,
                label_names,
            ),
            "folds": {
                fold_name: {
                    "num_cases": len(result.cases),
                    "confusion": result.global_confusion.tolist(),
                    "metrics": self._metrics_by_label(
                        fold_metrics[index],
                        label_names,
                    ),
                }
                for index, (fold_name, result) in enumerate(self.folds.items())
            },
        }

    def save_metrics_json(
        self,
        path: str | Path,
    ) -> None:
        with Path(path).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.metric_report(),
                file,
                indent=2,
                allow_nan=False,
            )

    def text_report(self) -> str:
        global_confusion = self.global_confusion
        global_metrics = self.global_metrics()
        fold_metrics = self.fold_metrics()
        mean, std, valid = aggregate_case_metrics(fold_metrics)
        label_names = self._label_names()

        lines = [
            "K-Fold Label Metrics Report",
            "===========================",
            "",
            f"Folds: {self.num_folds}",
            f"Labels: {len(self.labels)}",
            f"Total voxels: {int(global_confusion.sum().item())}",
            "",
            "Labels",
            "------",
        ]

        for index, name in enumerate(label_names):
            lines.append(f"{index}: {name}")

        lines.extend(
            [
                "",
                "Global confusion matrix",
                "-----------------------",
                "Rows: target labels",
                "Columns: prediction labels",
                "",
                self._format_confusion(global_confusion),
                "",
                "K-fold global metrics",
                "---------------------",
                ("Calculated from the sum of all fold confusion matrices."),
            ]
        )

        self._append_metrics(
            lines,
            global_metrics,
            label_names,
        )

        lines.extend(
            [
                "",
                "Metrics across folds",
                "--------------------",
                (
                    "Mean and population standard deviation "
                    "of fold-level global metrics."
                ),
                ("Undefined fold metrics are excluded from aggregation."),
            ]
        )

        self._append_summary(
            lines,
            mean,
            std,
            valid,
            label_names,
        )

        lines.extend(
            [
                "",
                "Per-fold global metrics",
                "-----------------------",
            ]
        )

        for index, (fold_name, result) in enumerate(self.folds.items()):
            lines.extend(
                [
                    "",
                    fold_name,
                    "~" * len(fold_name),
                    f"Cases: {len(result.cases)}",
                    (f"Voxels: {int(result.global_confusion.sum().item())}"),
                    "",
                    "Confusion matrix:",
                    self._format_confusion(result.global_confusion),
                ]
            )

            self._append_metrics(
                lines,
                fold_metrics[index],
                label_names,
            )

        return "\n".join(lines) + "\n"

    def save_text_report(
        self,
        path: str | Path,
    ) -> None:
        Path(path).write_text(
            self.text_report(),
            encoding="utf-8",
        )

    def _label_names(self) -> list:
        names = [""] * len(self.labels)

        for name, index in self.labels.items():
            names[index] = name

        return names

    @staticmethod
    def _metrics_by_label(
        metrics: Tensor,
        label_names: list[str],
    ) -> dict[str, Any]:
        return {
            label_name: {
                "label": label_index,
                **{
                    metric_name: _number(
                        metrics[
                            metric_index,
                            label_index,
                        ]
                    )
                    for metric_index, metric_name in enumerate(METRIC_NAMES)
                },
            }
            for label_index, label_name in enumerate(label_names)
        }

    def _summary_by_label(
        self,
        mean: Tensor,
        std: Tensor,
        valid: Tensor,
        label_names: list[str],
    ) -> dict[str, Any]:
        return {
            label_name: {
                "label": label_index,
                **{
                    metric_name: {
                        "mean": _number(
                            mean[
                                metric_index,
                                label_index,
                            ]
                        ),
                        "std": _number(
                            std[
                                metric_index,
                                label_index,
                            ]
                        ),
                        "valid_folds": int(
                            valid[
                                metric_index,
                                label_index,
                            ].item()
                        ),
                        "total_folds": self.num_folds,
                    }
                    for metric_index, metric_name in enumerate(METRIC_NAMES)
                },
            }
            for label_index, label_name in enumerate(label_names)
        }

    @staticmethod
    def _append_metrics(
        lines: list[str],
        metrics: Tensor,
        label_names: list[str],
    ) -> None:
        for label_index, label_name in enumerate(label_names):
            lines.extend(
                [
                    "",
                    f"{label_name} [{label_index}]",
                ]
            )

            for metric_index, metric_name in enumerate(METRIC_NAMES):
                value = metrics[
                    metric_index,
                    label_index,
                ]

                lines.append(f"  {metric_name}: {_format_number(value)}")

    def _append_summary(
        self,
        lines: list[str],
        mean: Tensor,
        std: Tensor,
        valid: Tensor,
        label_names: list[str],
    ) -> None:
        for label_index, label_name in enumerate(label_names):
            lines.extend(
                [
                    "",
                    f"{label_name} [{label_index}]",
                ]
            )

            for metric_index, metric_name in enumerate(METRIC_NAMES):
                valid_folds = int(
                    valid[
                        metric_index,
                        label_index,
                    ].item()
                )

                lines.append(
                    f"  {metric_name}: "
                    f"{_format_number(mean[metric_index, label_index])} "
                    f"+/- "
                    f"{_format_number(std[metric_index, label_index])} "
                    f"({valid_folds}/{self.num_folds} valid folds)"
                )

    @staticmethod
    def _format_confusion(
        confusion: Tensor,
    ) -> str:
        values = confusion.tolist()

        width = max(
            7,
            max(len(str(value)) for row in values for value in row) + 1,
        )

        lines = [
            "target".rjust(width)
            + "".join(str(index).rjust(width) for index in range(len(values)))
        ]

        for index, row in enumerate(values):
            lines.append(
                str(index).rjust(width)
                + "".join(str(value).rjust(width) for value in row)
            )

        return "\n".join(lines)


def _number(value: Tensor) -> float | None:
    result = float(value.item())

    if not math.isfinite(result):
        return None

    return result


def _format_number(value: Tensor) -> str:
    result = float(value.item())

    if not math.isfinite(result):
        return "N/A"

    return f"{result:.6f}"
