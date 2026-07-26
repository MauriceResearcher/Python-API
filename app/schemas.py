from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Frage zu den Python-Tutorial-Docs.",
        examples=["Wie funktioniert eine List Comprehension?"],
    )


class SourceChunk(BaseModel):
    header_1: Optional[str] = Field(None, description="Seitentitel")
    header_2: Optional[str] = Field(None, description="H2-Überschrift des Chunks")
    header_3: Optional[str] = Field(None, description="H3-Überschrift des Chunks, falls vorhanden")
    content_preview: str = Field(..., description="Erste 200 Zeichen des verwendeten Chunks")


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    ready: bool = Field(..., description="True sobald der RAG-Stack einsatzbereit ist")