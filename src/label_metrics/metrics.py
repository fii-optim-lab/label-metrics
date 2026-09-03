import torch
from torch import Tensor


METRIC_NAMES = (
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "accuracy",
)


def metrics_from_confusion(
    confusion: Tensor,
) -> Tensor:
    confusion = confusion.to(dtype=torch.float64)

    true_positive = confusion.diag()
    target_positive = confusion.sum(dim=1)
    predicted_positive = confusion.sum(dim=0)

    false_negative = target_positive - true_positive
    false_positive = predicted_positive - true_positive
    true_negative = confusion.sum() - true_positive - false_negative - false_positive

    return torch.stack(
        (
            _divide(
                2 * true_positive,
                2 * true_positive + false_positive + false_negative,
            ),
            _divide(
                true_positive,
                true_positive + false_positive + false_negative,
            ),
            _divide(
                true_positive,
                true_positive + false_positive,
            ),
            _divide(
                true_positive,
                true_positive + false_negative,
            ),
            _divide(
                true_negative,
                true_negative + false_positive,
            ),
            _divide(
                true_positive + true_negative,
                confusion.sum(),
            ),
        )
    )


def aggregate_case_metrics(
    case_metrics: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Aggregate metrics across cases.

    Parameters
    ----------
    case_metrics:
        Tensor with shape:
        (num_cases, num_metrics, num_labels)

    Returns
    -------
    mean:
        Mean across valid cases, shape:
        (num_metrics, num_labels)

    std:
        Population standard deviation across valid cases,
        shape:
        (num_metrics, num_labels)

    valid:
        Number of non-NaN cases, shape:
        (num_metrics, num_labels)
    """
    if case_metrics.ndim != 3:
        raise ValueError(
            "case_metrics must have shape (num_cases, num_metrics, num_labels)."
        )

    valid_mask = ~case_metrics.isnan()
    valid = valid_mask.sum(dim=0)

    values = case_metrics.nan_to_num(0.0)
    mean = values.sum(dim=0) / valid.clamp_min(1)

    differences = torch.where(
        valid_mask,
        case_metrics - mean,
        0.0,
    )

    variance = differences.square().sum(dim=0) / valid.clamp_min(1)
    std = variance.sqrt()

    no_values = valid == 0
    mean = mean.masked_fill(no_values, torch.nan)
    std = std.masked_fill(no_values, torch.nan)

    return mean, std, valid


def _divide(
    numerator: Tensor,
    denominator: Tensor,
) -> Tensor:
    result = torch.full_like(
        numerator,
        torch.nan,
        dtype=torch.float64,
    )

    valid = denominator != 0
    result[valid] = numerator[valid] / denominator[valid]

    return result
