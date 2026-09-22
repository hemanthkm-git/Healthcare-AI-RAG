from pathlib import Path
from pypdf import PdfReader


def read_text_file(file_path: str) -> str:
    """
    Read a text file and return its contents.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return path.read_text(encoding="utf-8")


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extract text from each PDF page and preserve page numbers.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        pages.append(
            {
                "document": path.name,
                "page_number": page_number,
                "text": page.extract_text() or "",
            }
        )

    return pages


if __name__ == "__main__":
    file_path = "data/raw/sample_clinical_note.pdf"

    pages = extract_text_from_pdf(file_path)

    print("PDF processed successfully!")
    print()

    for page in pages:
        print(f"--- Page {page['page_number']} ---")
        print(page["text"])