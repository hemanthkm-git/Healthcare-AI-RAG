from ollama import chat

from retriever import Retriever


MODEL_NAME = "llama3.2:3b"


def generate_answer(
    question: str,
    context: str,
) -> str:
    """
    Generate an answer using retrieved clinical context.
    """

    prompt = f"""
You are a healthcare documentation assistant.

Use ONLY the clinical documentation provided
in the context.

Do not invent, assume, or add clinical facts.

If the answer is not supported by the context,
say:

"I could not find enough information in the
provided clinical documentation."

Clinical Context:
-----------------
{context}
-----------------

Question:
{question}

Answer:
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]


def build_context(results: list[dict]) -> str:
    """
    Combine retrieved chunks into context for the LLM.
    """

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
Document: {result['document']}
Page: {result['page_number']}
Chunk: {result['chunk_id']}

{result['text']}
"""
        )

    return "\n".join(context_parts)


if __name__ == "__main__":

    retriever = Retriever(
        "data/processed/sample_clinical_note.json"
    )

    question = "What diagnosis does the patient have?"

    results = retriever.retrieve(
        question,
        top_k=3,
    )

    context = build_context(
        results
    )

    answer = generate_answer(
        question,
        context,
    )

    print()
    print("=============================")
    print("QUESTION")
    print("=============================")
    print(question)

    print()
    print("=============================")
    print("RETRIEVED CONTEXT")
    print("=============================")
    print(context)

    print()
    print("=============================")
    print("AI ANSWER")
    print("=============================")
    print(answer)