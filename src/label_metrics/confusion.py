from torch import Tensor


def py_confusion_matrix(
    prediction: Tensor,
    target: Tensor,
    num_labels: int,
) -> Tensor:
    return (
        prediction.add_(target, alpha=num_labels)
        .view(-1)
        .bincount(
            minlength=num_labels * num_labels,
        )
        .reshape(num_labels, num_labels)
    )
