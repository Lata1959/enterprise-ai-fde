from datetime import datetime, timezone
from fastapi import FastAPI, Request
from app.services.ai_service import ai_service
from pydantic import BaseModel
from app.core.config import settings
from app.api.routes.knowledge import router as knowledge_router

class ChatRequest(BaseModel):
    question: str


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Enterprise AI assistant with "
        "SAP and Salesforce integrations"
    ),
)


app.include_router(knowledge_router)


@app.get("/")
async def root(request: Request):
    return {
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "running",
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.OPENAI_MODEL,
    }

@app.post("/chat")
async def chat(request: ChatRequest):

    answer = ai_service.ask(request.question)

    return {
        "question": request.question,
        "answer": answer
    }