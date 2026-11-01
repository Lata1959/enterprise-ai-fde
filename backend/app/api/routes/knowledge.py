from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.rag_service import rag_service


router = APIRouter(
    prefix="/knowledge",
    tags=["Enterprise Knowledge"]
)


class KnowledgeQuestion(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10
    )


@router.post("/ask")
async def ask_knowledge_base(
    request: KnowledgeQuestion
):

    result = rag_service.answer_question(
        question=request.question,
        top_k=request.top_k
    )

    return result