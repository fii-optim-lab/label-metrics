from dataclasses import dataclass
from pathlib import Path

from tqdm.contrib.concurrent import process_map

from label_metrics.confusion import py_confusion_matrix
from label_metrics.loading import load_case
from label_metrics.readers.base import BaseReader
from label_metrics.readers.factory import (
    file_type,
    get_reader,
)
from label_metrics.result import (
    CaseResult,
    EvaluationResult,
)


@dataclass(frozen=True)
class EvaluationTask:
    case_name: str
    prediction_path: Path
    target_path: Path
    num_labels: int
    prediction_reader: BaseReader
    target_reader: BaseReader


def evaluate_folders(
    prediction_folder: str | Path,
    ground_truth_folder: str | Path,
    labels: dict[str, int],
    *,
    prediction_prefix: str = "",
    prediction_suffix: str,
    ground_truth_prefix: str = "",
    ground_truth_suffix: str | None = None,
    reader: BaseReader | None = None,
    reorient: bool = False,
    workers: int | None = None,
    chunksize: int = 1,
) -> EvaluationResult:
    _validate_labels(labels)

    if not prediction_suffix:
        raise ValueError("prediction_suffix cannot be empty.")

    if ground_truth_suffix is None:
        ground_truth_suffix = prediction_suffix

    if not ground_truth_suffix:
        raise ValueError("ground_truth_suffix cannot be empty.")

    prediction_folder = Path(prediction_folder)
    ground_truth_folder = Path(ground_truth_folder)

    prediction_paths = _find_files(
        prediction_folder,
        prefix=prediction_prefix,
        suffix=prediction_suffix,
    )

    if not prediction_paths:
        raise FileNotFoundError(
            "No prediction files matching "
            f"{prediction_prefix!r} + <case> + "
            f"{prediction_suffix!r} were found in "
            f"{str(prediction_folder)!r}."
        )

    predictions = _index_cases(
        prediction_paths,
        prefix=prediction_prefix,
        suffix=prediction_suffix,
        kind="prediction",
    )

    ground_truth_paths = _find_files(
        ground_truth_folder,
        prefix=ground_truth_prefix,
        suffix=ground_truth_suffix,
    )

    ground_truths = _index_cases(
        ground_truth_paths,
        prefix=ground_truth_prefix,
        suffix=ground_truth_suffix,
        kind="ground-truth",
    )

    missing = sorted(set(predictions) - set(ground_truths))

    if missing:
        names = ", ".join(repr(case_name) for case_name in missing)

        raise FileNotFoundError(f"Missing ground-truth files for cases: {names}.")

    prediction_readers = _resolve_readers(
        predictions.values(),
        reader=reader,
        reorient=reorient,
    )
    target_readers = _resolve_readers(
        (ground_truths[case_name] for case_name in predictions),
        reader=reader,
        reorient=reorient,
    )

    tasks = [
        EvaluationTask(
            case_name=case_name,
            prediction_path=predictions[case_name],
            target_path=ground_truths[case_name],
            num_labels=len(labels),
            prediction_reader=prediction_readers[file_type(predictions[case_name])],
            target_reader=target_readers[file_type(ground_truths[case_name])],
        )
        for case_name in sorted(predictions)
    ]

    cases = process_map(
        _evaluate_task,
        tasks,
        max_workers=workers,
        chunksize=chunksize,
        desc="Evaluating",
    )

    return EvaluationResult(
        labels=dict(labels),
        cases=tuple(cases),
    )


def _evaluate_task(
    task: EvaluationTask,
) -> CaseResult:
    loaded = load_case(
        prediction_path=task.prediction_path,
        target_path=task.target_path,
        case_name=task.case_name,
        num_labels=task.num_labels,
        prediction_reader=task.prediction_reader,
        target_reader=task.target_reader,
    )

    confusion = py_confusion_matrix(
        loaded.prediction,
        loaded.target,
        task.num_labels,
    )

    return CaseResult(
        case_name=task.case_name,
        prediction_shape=loaded.prediction_shape,
        target_shape=loaded.target_shape,
        confusion=confusion,
    )


def _find_files(
    folder: Path,
    *,
    prefix: str,
    suffix: str,
) -> list:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {str(folder)!r}.")

    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {str(folder)!r}.")

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.name.startswith(prefix)
        and path.name.endswith(suffix)
    )


def _index_cases(
    paths: list[Path],
    *,
    prefix: str,
    suffix: str,
    kind: str,
) -> dict[str, Path]:
    cases: dict[str, Path] = {}

    for path in paths:
        case_name = _case_name(
            path,
            prefix=prefix,
            suffix=suffix,
        )

        if not case_name:
            raise ValueError(f"Cannot derive a case name from {str(path)!r}.")

        previous = cases.get(case_name)

        if previous is not None:
            raise ValueError(
                f"Duplicate {kind} case name "
                f"{case_name!r}: "
                f"{str(previous)!r} and "
                f"{str(path)!r}."
            )

        cases[case_name] = path

    return cases


def _case_name(
    path: Path,
    *,
    prefix: str,
    suffix: str,
) -> str:
    name = path.name

    if prefix:
        name = name[len(prefix) :]

    if suffix:
        name = name[: -len(suffix)]

    return name


def _resolve_readers(
    paths,
    *,
    reader: BaseReader | None,
    reorient: bool,
) -> dict[str, BaseReader]:
    readers: dict[str, BaseReader] = {}

    for path in paths:
        suffix = file_type(path)

        if suffix in readers:
            continue

        if reader is not None:
            accepted = reader.accepted_file_types

            if accepted and suffix not in accepted:
                raise ValueError(
                    f"Reader {type(reader).__name__} does not "
                    f"support file type {suffix!r} for file "
                    f"{str(path)!r}."
                )

            readers[suffix] = reader
        else:
            readers[suffix] = get_reader(
                path,
                reorient=reorient,
            )

    return readers


def _validate_labels(
    labels: dict[str, int],
) -> None:
    if not isinstance(labels, dict):
        raise TypeError("labels must be a dictionary.")

    if not labels:
        raise ValueError("The label dictionary cannot be empty.")

    for name, index in labels.items():
        if not isinstance(name, str):
            raise TypeError("Every label name must be a string.")

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"Label {name!r} must map to an integer.")

    actual = sorted(labels.values())
    expected = list(range(len(labels)))

    if actual != expected:
        raise ValueError(
            "Label dictionary values must be exactly "
            f"0 through {len(labels) - 1}. "
            f"Got {actual}."
        )
