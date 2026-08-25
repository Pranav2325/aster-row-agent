import math
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = "models/gemini-embedding-001"

def cosine_similarity(a, b):
    """Measures how similar two number-lists are, from -1 to 1."""
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))
    return dot_product / (magnitude_a * magnitude_b)


# Quick sanity tests with tiny, easy-to-predict numbers
if __name__ == "__main__":
    identical_a = [1, 0, 0]
    identical_b = [1, 0, 0]
    print("Identical vectors (expect ~1.0):", cosine_similarity(identical_a, identical_b))

    opposite_a = [1, 0, 0]
    opposite_b = [-1, 0, 0]
    print("Opposite vectors (expect ~-1.0):", cosine_similarity(opposite_a, opposite_b))

    unrelated_a = [1, 0, 0]
    unrelated_b = [0, 1, 0]
    print("Perpendicular vectors (expect ~0.0):", cosine_similarity(unrelated_a, unrelated_b))


def load_index():
    with open("index.json","r",encoding="utf-8") as f:
        return json.load(f)
    
def embed_text(text):
    result = genai.embed_content(model=EMBED_MODEL, content=text)
    return result["embedding"]

def search(query, top_k=3):
    chunks = load_index()
    query_embedding = embed_text(query)

    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])

        # Penalize non-active documents heavily instead of trusting raw similarity
        if chunk["status"] != "active":
            score -= 0.5  # big penalty pushes them down, doesn't fully hide them

        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

if __name__ == "__main__":
    question = "How many days do I have to return an item?"
    results = search(question)

    print(f"Question: {question}\n")
    for score, chunk in results:
        print(f"Score: {score:.3f} | Status: {chunk['status']} | File: {chunk['source_file']}")
        print(chunk["text"][:150])
        print()