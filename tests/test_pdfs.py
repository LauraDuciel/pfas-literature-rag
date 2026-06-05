import hashlib

from pfas_lit_rag.pdfs import file_sha256


def test_file_sha256_depends_on_content(tmp_path) -> None:
    path = tmp_path / "document.pdf"
    content = b"pdf bytes"
    path.write_bytes(content)

    assert file_sha256(path) == hashlib.sha256(content).hexdigest()
