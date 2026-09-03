from typing import Protocol

from numpy import ndarray


class Reader(Protocol):
    accepted_file_types: tuple[str, ...]

    def read(self, path: str) -> ndarray:
        ...