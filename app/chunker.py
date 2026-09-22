def create_chunks(
    text: str,
    document_name: str,
    page_number: int,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[dict]:
    """
    Split text into overlapping chunks while preserving
    document and page metadata.
    """

    words = text.split()

    chunks = []

    start = 0
    chunk_id = 1

    while start < len(words):

        end = start + chunk_size

        chunk_text = " ".join(words[start:end])

        chunks.append(
            {
                "document": document_name,
                "page_number": page_number,
                "chunk_id": chunk_id,
                "text": chunk_text,
            }
        )

        start += chunk_size - overlap
        chunk_id += 1

    return chunks


if __name__ == "__main__":

    # Sample clinical text for testing
    sample_text = """
    Clinical Note

    Date: 09/15/2026

    Chief Complaint:
    Patient presents for follow-up of hypertension and type 2 diabetes mellitus.

    History:
    The patient has a history of hypertension and type 2 diabetes mellitus.
    Current medications include metformin and lisinopril.

    Assessment:
    1. Essential hypertension.
    2. Type 2 diabetes mellitus without complications.

    Plan:
    Continue current medications.
    Follow up in three months.
    """

    chunks = create_chunks(
        text=sample_text,
        document_name="sample_clinical_note.pdf",
        page_number=1,
        chunk_size=30,
        overlap=5,
    )

    print(f"Total chunks created: {len(chunks)}")

    for chunk in chunks:

        print()
        print("-----------------------------")
        print(f"Document: {chunk['document']}")
        print(f"Page: {chunk['page_number']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print("-----------------------------")
        print(chunk["text"])