import json

from document_processor import extract_text_from_pdf
from text_cleaner import clean_text
from chunker import create_chunks


def process_pdf(file_path: str) -> list[dict]:
    """
    Complete PDF ingestion pipeline.

    PDF
    ↓
    Text extraction
    ↓
    Text cleaning
    ↓
    Chunking
    """

    pages = extract_text_from_pdf(file_path)

    all_chunks = []

    for page in pages:

        cleaned_text = clean_text(page["text"])

        if not cleaned_text:
            continue

        chunks = create_chunks(
            text=cleaned_text,
            document_name=page["document"],
            page_number=page["page_number"],
        )

        all_chunks.extend(chunks)

    return all_chunks


def save_chunks(chunks: list[dict], output_file: str) -> None:
    """
    Save processed chunks to a JSON file.
    """

    with open(output_file, "w", encoding="utf-8") as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":

    input_file = "data/raw/sample_clinical_note.pdf"

    output_file = "data/processed/sample_clinical_note.json"

    chunks = process_pdf(input_file)

    save_chunks(
        chunks,
        output_file,
    )

    print("Document ingestion completed!")
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved to: {output_file}")