from app.core.config import settings
from openai import OpenAI


class AIService:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY
        )
        self.model = settings.OPENAI_MODEL

    def ask(self, question: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=question,
        )
        return response.output_text


ai_service = AIService()