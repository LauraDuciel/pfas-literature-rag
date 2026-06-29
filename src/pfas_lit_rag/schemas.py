from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl


class DocumentPage(BaseModel):
    document_id: str
    title: str
    source_path: Path
    page_number: int
    text: str


class DocumentRecord(BaseModel):
    document_id: str
    title: str
    source_path: str
    file_sha256: str
    pages_extracted: int
    chunks_indexed: int


class TextChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source_path: str
    page_start: int
    page_end: int
    text: str


class SearchResult(BaseModel):
    chunk: TextChunk
    score: float

    @property
    def citation(self) -> str:
        if self.chunk.page_start == self.chunk.page_end:
            pages = f"p. {self.chunk.page_start}"
        else:
            pages = f"pp. {self.chunk.page_start}-{self.chunk.page_end}"
        return f"{self.chunk.title}, {pages}"


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    retrieved: list[SearchResult]


class SearchRequest(BaseModel):
    query: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class LiteratureRecord(BaseModel):
    title: str
    source: str
    pdf_url: HttpUrl
    landing_url: HttpUrl | None = None
    doi: str | None = None
    openalex_id: str | None = None
    publication_year: int | None = None
    publication_date: str | None = None
    journal: str | None = None
    publisher: str | None = None
    cited_by_count: int | None = None
    work_type: str | None = None
    is_open_access: bool | None = None
    open_access_status: str | None = None
    open_access_version: str | None = None
    license: str | None = None
    authors: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    downloaded_path: str | None = None
