import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(file_path: str) -> list[dict]:
    """
    Load processed document chunks from JSON.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def create_embeddings(
    model,
    chunks: list[dict],
):
    """
    Create embeddings for all document chunks.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
    )

    return embeddings


def create_faiss_index(embeddings):
    """
    Create a FAISS vector index.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        embeddings.astype("float32")
    )

    return index

def search(
    model,
    index,
    chunks: list[dict],
    query: str,
    top_k: int = 3,
):
    """
    Search the vector database for the most
    relevant document chunks.
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    )

    distances, indices = index.search(
        query_embedding.astype("float32"),
        top_k,
    )

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0],
    ):

        if index_number == -1:
            continue

        result = chunks[index_number].copy()

        result["distance"] = float(distance)

        results.append(result)

    return results

if __name__ == "__main__":

    model = SentenceTransformer(MODEL_NAME)

    chunks = load_chunks(
        "data/processed/sample_clinical_note.json"
    )

    embeddings = create_embeddings(
        model,
        chunks,
    )

    index = create_faiss_index(
        embeddings
    )

    query = "What diagnosis does the patient have?"

    results = search(
        model=model,
        index=index,
        chunks=chunks,
        query=query,
        top_k=3,
    )

    print()
    print("Search results:")
    print()

    for result in results:

        print("-----------------------------")
        print(
            f"Document: {result['document']}"
        )
        print(
            f"Page: {result['page_number']}"
        )
        print(
            f"Chunk: {result['chunk_id']}"
        )
        print(
            f"Distance: {result['distance']}"
        )
        print("-----------------------------")
        print(result["text"])