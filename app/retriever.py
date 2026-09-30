from sentence_transformers import SentenceTransformer

from vector_store import (
    load_chunks,
    create_embeddings,
    create_faiss_index,
    search,
)


MODEL_NAME = "all-MiniLM-L6-v2"


class Retriever:
    """
    Retrieve the most relevant clinical document
    chunks for a user query.
    """

    def __init__(
        self,
        chunks_file: str,
    ):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print("Loading clinical chunks...")

        self.chunks = load_chunks(
            chunks_file
        )

        print("Creating embeddings...")

        embeddings = create_embeddings(
            self.model,
            self.chunks,
        )

        print("Creating FAISS index...")

        self.index = create_faiss_index(
            embeddings
        )

        print("Retriever ready!")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Retrieve the most relevant clinical
        chunks for the query.
        """

        results = search(
            model=self.model,
            index=self.index,
            chunks=self.chunks,
            query=query,
            top_k=top_k,
        )

        return results


if __name__ == "__main__":

    retriever = Retriever(
        "data/processed/sample_clinical_note.json"
    )

    query = "What diagnosis does the patient have?"

    results = retriever.retrieve(
        query,
        top_k=3,
    )

    print()
    print("Query:")
    print(query)

    print()
    print("Retrieved clinical evidence:")
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
        print()