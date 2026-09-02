from pathlib import Path

import pytest

from solution.ex2 import main

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_full_dataset_reproduces_reference_output(tmp_path) -> None:
    output_path = tmp_path / "output.txt"

    main(
        REPOSITORY_ROOT / "data" / "develop.txt",
        REPOSITORY_ROOT / "data" / "test.txt",
        "honduras",
        output_path,
    )

    expected_lines = (
        (REPOSITORY_ROOT / "results" / "output.txt").read_text(encoding="utf-8").splitlines()
    )
    actual_lines = output_path.read_text(encoding="utf-8").splitlines()

    assert actual_lines[:5] == expected_lines[:5]
    assert len(actual_lines) == len(expected_lines)

    for actual_line, expected_line in zip(actual_lines[5:], expected_lines[5:], strict=True):
        actual_key, actual_value = actual_line.split("\t", maxsplit=1)
        expected_key, expected_value = expected_line.split("\t", maxsplit=1)
        assert actual_key == expected_key
        assert float(actual_value) == pytest.approx(float(expected_value), rel=1e-10)
