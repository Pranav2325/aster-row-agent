import json
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai
from chunking import KB_FOLDER, load_document, chunk_body

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "models/gemini-embedding-001"

def build_all_chunks():
    """Reuses chunking.py logic to build the list of all 60 chunk dicts."""
    files = os.listdir(KB_FOLDER)
    all_chunks = []
    for filename in files:
        metadata, body = load_document(filename)
        for chunk_text in chunk_body(body):
            if len(chunk_text) < 30:
                continue
            all_chunks.append({
                "text": chunk_text,
                "source_file": filename,
                "status": metadata.get("status", "unknown"),
                "document_id": metadata.get("document_id", "unknown"),
            })
    return all_chunks

def embed_text(text):
    """Calls Gemini to turn one piece of text into an embedding."""
    result = genai.embed_content(model=EMBED_MODEL, content=text)
    return result["embedding"]

if __name__ == "__main__":
    chunks = build_all_chunks()
    print(f"Embedding {len(chunks)} chunks... this will take a minute.\n")

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embed_text(chunk["text"])
        print(f"[{i+1}/{len(chunks)}] embedded: {chunk['source_file']}")
        time.sleep(0.5) 

    with open("index.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    print("\nDone! Saved to index.json")
    
    