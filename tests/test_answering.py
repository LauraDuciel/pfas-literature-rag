from pfas_lit_rag.answering import _ensure_sources_section


def test_ensure_sources_section_appends_citations() -> None:
    answer = _ensure_sources_section("PFAS can be measured by LC-MS [1].", ["Paper A, p. 2"])

    assert "Sources:" in answer
    assert "[1] Paper A, p. 2" in answer


def test_ensure_sources_section_does_not_duplicate_sources() -> None:
    answer = _ensure_sources_section("Text.\n\nSources:\n[1] Paper A, p. 2", ["Paper A, p. 2"])

    assert answer.count("Sources:") == 1
