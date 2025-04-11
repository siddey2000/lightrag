import asyncio  # We'll use asyncio for the ollama calls
from utils import read_pdf, chunk_text, generate_embeddings, ollama_model_complete
from config import PDF_FILE_PATH, CHUNK_SIZE, CHUNK_OVERLAP, OLLAMA_LLM_MODEL, OLLAMA_EMBEDDING_MODEL
import chromadb
import numpy as np


async def main():
    pdf_text = read_pdf()  # Use the default file path from config

    if not pdf_text:
        print("Failed to load the PDF content.")
        return

    text_chunks, chunk_ids = chunk_text(pdf_text)
    print(f"Number of text chunks: {len(text_chunks)}")

    # Generate embeddings
    text_embeddings = generate_embeddings(text_chunks, model_name=OLLAMA_EMBEDDING_MODEL)

    # Initialize ChromaDB client and collection
    client = chromadb.Client()
    collection = client.create_collection("my_rag_collection")

    # Add chunks and embeddings to ChromaDB
    collection.add(
        ids=[str(i) for i in chunk_ids],  # Convert IDs to strings
        documents=text_chunks,
        embeddings=text_embeddings.tolist(),  # Convert embeddings to lists
    )

    # RAG Query Example
    query = "What are the recommended PSUs for RTX 4080?"
    query_embedding = generate_embeddings([query], model_name=OLLAMA_EMBEDDING_MODEL)[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,  # Number of relevant chunks to retrieve
    )

    context = "\n".join(results["documents"][0])  # Combine retrieved chunks
    prompt = f"Context:\n{context}\n\nUser Question: {query}"
    response = await ollama_model_complete(prompt=prompt, model_name=OLLAMA_LLM_MODEL)
    print(f"LLM Response:\n{response}")


if __name__ == "__main__":
    asyncio.run(main())