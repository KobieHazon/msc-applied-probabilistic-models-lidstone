"""Helper to write to the output file in the format of the exercise"""

import os
from pathlib import Path
from types import TracebackType
from typing import Any


class OutputFileWriter:
    """
    Helper class for writing to the output file in the specified format
    """

    def __init__(self, output_file_name: str | Path):
        self._output_file = open(output_file_name, "w", encoding="utf-8")
        self._output_counter = 0
        self._prepare_output_file()  # prepare writing to it by writing the author's name

    def write(self, to_write: Any):
        """
        Write a row to the output file
        :param to_write: value of the output row, must be convertable to str
        """
        try:
            to_write = str(to_write)
        except TypeError as exc:
            raise TypeError(f"can't write type {type(to_write)} to text file") from exc

        self._output_file.write(
            f"#Output{self._output_counter}\t{to_write}\n"
        )  # format asked for by the exercise
        self._output_counter += 1

    def _prepare_output_file(self):
        """
        Prepare the output file for writing output values to it
        """
        self._output_file.write("#Students\tKobie Hazon\n")
        self._output_counter += 1

    def close(self) -> None:
        """Flush and close the output file."""
        self._output_file.close()

    def __enter__(self) -> "OutputFileWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def file_name(self) -> str:
        return os.path.basename(
            self._output_file.name
        )  # instructed to only write the base name of the file
