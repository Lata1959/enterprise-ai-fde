from pathlib import Path
from app.ingestion.document_loader import load_document
from app.services.rag_service import rag_service


PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"


def ingest_documents():

    for file_path in KNOWLEDGE_DIR.iterdir():

        if file_path.suffix.lower() not in [".txt", ".pdf"]:
            continue

        print(f"Processing: {file_path.name}")

        content = load_document(str(file_path))

        chunks = rag_service.index_document(
            text=content,
            source=file_path.name
        )

        print(f"Indexed {chunks} chunks")


if __name__ == "__main__":
    ingest_documents()