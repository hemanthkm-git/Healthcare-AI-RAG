import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving
    the actual clinical content.
    """

    # Replace multiple spaces or tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove unnecessary leading/trailing whitespace
    text = text.strip()

    return text


if __name__ == "__main__":

    sample_text = """
    Clinical Note


    Patient presents with hypertension.


    Assessment:
    Essential hypertension.
    """

    cleaned_text = clean_text(sample_text)

    print("Cleaned text:")
    print(cleaned_text)