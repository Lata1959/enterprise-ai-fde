from app.services.rag_service import rag_service


question = (
     "Which equipment should not be assigned "
    "to an active construction project?"
)

results = rag_service.search(question, top_k=3)

print("\nRETRIEVED KNOWLEDGE\n")

for result in results:

    print("Source:", result["source"])
    print("Distance:", result["distance"])
    print("Content:", result["content"])
    print("-" * 60)