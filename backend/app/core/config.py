from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: <repo>/
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    APP_NAME: str = "Enterprise AI FDE Application"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"

    model_name: str = "gpt-4o"
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    model_temperature: float = 0.0
    model_max_tokens: int = 2048
    model_top_p: float = 1.0
    model_frequency_penalty: float = 0.0
    model_presence_penalty: float = 0.0

    SAP_API_KEY: str = ""
    SALESFORCE_CLIENT_ID: str = ""
    SALESFORCE_CLIENT_SECRET: str = ""
    OPENAI_API_KEY: str = ""

    database_url: str = "sqlite:///./app.db"
    log_level: str = "INFO"

    CHROMA_DB_PATH: str = str(
        PROJECT_ROOT / "data" / "chroma"
    )   

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()