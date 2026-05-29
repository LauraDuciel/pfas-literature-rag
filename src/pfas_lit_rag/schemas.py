from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl


class DocumentPage(BaseModel):
    document_id: str
    title: str
    source_path: Path
    page_number: int
    text: str


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
    publication_year: int | None = None
    license: str | None = None
    downloaded_path: str | None = None
