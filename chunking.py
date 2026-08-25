import os
import re

KB_FOLDER = "assignment-data/knowledge-base"


def load_document(filename):
    """Reads one file, returns its metadata (dict) and body (text)."""
    file_path = os.path.join(KB_FOLDER, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = content.split("---")
    front_matter_raw = parts[1].strip()
    body = parts[2].strip()

    metadata = {}
    for line in front_matter_raw.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    return metadata, body


def chunk_body(body):
    """Splits body text into chunks, one per ## heading section."""
    sections = re.split(r"\n(?=## )", body)
    return [s.strip() for s in sections if s.strip()]


if __name__ == "__main__":
    files = os.listdir(KB_FOLDER)
    documents = []

    for filename in files:
        metadata, body = load_document(filename)
        documents.append({"filename": filename, "metadata": metadata, "body": body})

    for doc in documents:
        print(doc["filename"], "->", doc["metadata"]["status"])

    all_chunks = []
    for doc in documents:
        chunks = chunk_body(doc["body"])
        for chunk_text in chunks:
            if len(chunk_text) < 30:
                continue
            all_chunks.append({
                "text": chunk_text,
                "source_file": doc["filename"],
                "status": doc["metadata"].get("status", "unknown"),
                "document_id": doc["metadata"].get("document_id", "unknown"),
            })

    print(f"Total chunks created across all documents: {len(all_chunks)}\n")
    for c in all_chunks[:5]:
        print(f"[{c['status']}] {c['source_file']} -> {c['text'][:60]}...")