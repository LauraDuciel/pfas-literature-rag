from pathlib import Path

import pytest
import typer

from pfas_lit_rag.cli.audit import _read_answer


def test_read_answer_from_text() -> None:
    assert _read_answer(answer="answer", answer_file=None) == "answer"


def test_read_answer_from_file(tmp_path: Path) -> None:
    path = tmp_path / "answer.txt"
    path.write_text("saved answer", encoding="utf-8")

    assert _read_answer(answer=None, answer_file=path) == "saved answer"


def test_read_answer_rejects_missing_input() -> None:
    with pytest.raises(typer.BadParameter):
        _read_answer(answer=None, answer_file=None)
