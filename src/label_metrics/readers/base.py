from abc import ABC, abstractmethod
from pathlib import Path

from numpy import ndarray


class BaseReader(ABC):
    accepted_file_types: tuple[str, ...] = ()

    @abstractmethod
    def read(self, path: str | Path) -> ndarray:
        pass
