from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_root: Path = Path.cwd()
    raw_pdf_dir: Path = Path("data/raw_pdfs")
    index_dir: Path = Path("data/index")
    metadata_dir: Path = Path("data/metadata")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"

    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 5

    request_timeout_seconds: float = 60.0
    collector_user_agent: str = "pfas-lit-rag/0.1"

    model_config = SettingsConfigDict(env_prefix="PFAS_RAG_", env_file=".env", extra="ignore")

    def resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def resolved_raw_pdf_dir(self) -> Path:
        return self.resolve_path(self.raw_pdf_dir)

    @property
    def resolved_index_dir(self) -> Path:
        return self.resolve_path(self.index_dir)

    @property
    def resolved_metadata_dir(self) -> Path:
        return self.resolve_path(self.metadata_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
