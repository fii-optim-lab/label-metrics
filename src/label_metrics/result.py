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


@dataclass(frozen=True)
class CaseResult:
    case_name: str
    prediction_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    confusion: Tensor

    @property
    def voxel_count(self) -> int:
        return int(self.confusion.sum().item())


@dataclass(frozen=True)
class EvaluationResult:
    labels: dict[str, int]
    cases: tuple[CaseResult, ...]

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("EvaluationResult must contain at least one case.")

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def global_confusion(self) -> Tensor:
        confusion = torch.zeros_like(self.cases[0].confusion)

        for case in self.cases:
            confusion += case.confusion

        return confusion

    @property
    def total_voxel_count(self) -> int:
        return int(self.global_confusion.sum().item())

    def global_metrics(self) -> Tensor:
        return metrics_from_confusion(self.global_confusion)

    def case_metrics(self) -> Tensor:
        return torch.stack(
            [metrics_from_confusion(case.confusion) for case in self.cases]
        )

    def case_metric_summary(
        self,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return aggregate_case_metrics(self.case_metrics())

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": dict(self.labels),
            "cases": [
                {
                    "case_name": case.case_name,
                    "prediction_shape": list(case.prediction_shape),
                    "target_shape": list(case.target_shape),
                    "confusion": case.confusion.tolist(),
                }
                for case in self.cases
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> EvaluationResult:
        return cls(
            labels={str(name): int(index) for name, index in data["labels"].items()},
            cases=tuple(
                CaseResult(
                    case_name=str(case["case_name"]),
                    prediction_shape=tuple(case["prediction_shape"]),
                    target_shape=tuple(case["target_shape"]),
                    confusion=torch.tensor(
                        case["confusion"],
                        dtype=torch.int64,
                    ),
                )
                for case in data["cases"]
            ),
        )

    def serialize(
        self,
        path: str | Path,
    ) -> None:
        with Path(path).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
                allow_nan=False,
            )

    @classmethod
    def deserialize(
        cls,
        path: str | Path,
    ) -> EvaluationResult:
        with Path(path).open(
            encoding="utf-8",
        ) as file:
            return cls.from_dict(json.load(file))

    def metric_report(self) -> dict[str, Any]:
        global_metrics = self.global_metrics()
        case_metrics = self.case_metrics()
        mean, std, valid = aggregate_case_metrics(case_metrics)

        label_names = self._label_names()

        report = {
            "labels": dict(self.labels),
            "num_cases": len(self.cases),
            "total_voxels": self.total_voxel_count,
            "global_confusion": self.global_confusion,
            "global_metrics": self._metrics_by_label(
                global_metrics,
                label_names,
            ),
            "case_summary": self._summary_by_label(
                mean,
                std,
                valid,
                label_names,
            ),
            "cases": {
                case.case_name: {
                    "prediction_shape": list(case.prediction_shape),
                    "target_shape": list(case.target_shape),
                    "confusion": case.confusion,
                    "metrics": self._metrics_by_label(
                        case_metrics[index],
                        label_names,
                    ),
                }
                for index, case in enumerate(self.cases)
            },
        }

        return _json_value(report)

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
        global_metrics = metrics_from_confusion(global_confusion)
        case_metrics = self.case_metrics()
        mean, std, valid = aggregate_case_metrics(case_metrics)
        label_names = self._label_names()

        lines = [
            "Label Metrics Report",
            "====================",
            "",
            f"Cases: {len(self.cases)}",
            f"Labels: {self.num_labels}",
            f"Total voxels: {self.total_voxel_count}",
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
                "Global metrics",
                "--------------",
                ("Calculated from the sum of all case confusion matrices."),
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
                "Per-case metric summary",
                "-----------------------",
                ("Mean and population standard deviation across valid cases."),
                ("Undefined metric values are excluded from aggregation."),
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
                "Per-case metrics",
                "----------------",
            ]
        )

        for case_index, case in enumerate(self.cases):
            lines.extend(
                [
                    "",
                    case.case_name,
                    "~" * len(case.case_name),
                    (f"Prediction shape: {case.prediction_shape}"),
                    (f"Ground-truth shape: {case.target_shape}"),
                    f"Voxels: {case.voxel_count}",
                ]
            )

            self._append_metrics(
                lines,
                case_metrics[case_index],
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
        names = [""] * self.num_labels

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

    @staticmethod
    def _summary_by_label(
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
                        "valid_cases": int(
                            valid[
                                metric_index,
                                label_index,
                            ].item()
                        ),
                    }
                    for metric_index, metric_name in enumerate(METRIC_NAMES)
                },
            }
            for label_index, label_name in enumerate(label_names)
        }

    @staticmethod
    def _format_confusion(
        confusion: Tensor,
    ) -> str:
        values = confusion.tolist()
        width = max(
            7,
            max(len(str(value)) for row in values for value in row) + 1,
        )

        header = "target".rjust(width) + "".join(
            str(index).rjust(width) for index in range(len(values))
        )

        rows = [header]

        for index, row in enumerate(values):
            rows.append(
                str(index).rjust(width)
                + "".join(str(value).rjust(width) for value in row)
            )

        return "\n".join(rows)

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
        total_cases = len(self.cases)

        for label_index, label_name in enumerate(label_names):
            lines.extend(
                [
                    "",
                    f"{label_name} [{label_index}]",
                ]
            )

            for metric_index, metric_name in enumerate(METRIC_NAMES):
                mean_value = mean[
                    metric_index,
                    label_index,
                ]
                std_value = std[
                    metric_index,
                    label_index,
                ]
                valid_cases = int(
                    valid[
                        metric_index,
                        label_index,
                    ].item()
                )

                lines.append(
                    f"  {metric_name}: "
                    f"{_format_number(mean_value)} "
                    f"+/- {_format_number(std_value)} "
                    f"({valid_cases}/{total_cases} valid cases)"
                )


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


def _json_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return _json_value(value.tolist())

    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value
