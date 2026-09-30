import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    model = SentenceTransformer(MODEL_NAME)
    return model


def create_embedding(model, text: str):
    embedding = model.encode(text)
    return embedding


def cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors.
    """

    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    similarity = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )

    return similarity


if __name__ == "__main__":

    model = load_embedding_model()

    text1 = "The patient has hypertension."
    text2 = "The patient has high blood pressure."
    text3 = "The patient has a fractured leg."

    embedding1 = create_embedding(model, text1)
    embedding2 = create_embedding(model, text2)
    embedding3 = create_embedding(model, text3)

    similarity_1_2 = cosine_similarity(
        embedding1,
        embedding2,
    )

    similarity_1_3 = cosine_similarity(
        embedding1,
        embedding3,
    )

    print("Similarity between sentence 1 and 2:")
    print(similarity_1_2)

    print()
    print("Similarity between sentence 1 and 3:")
    print(similarity_1_3)