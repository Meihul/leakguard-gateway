import os
import re
from pathlib import Path

import chromadb

from gateway.detectors.injection import detect_injection

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "data" / "docs"
DEFAULT_DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "docs"
CHUNK_SIZE = 500


def _parse_doc(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    access = None
    title = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ACCESS:"):
            access = line.split(":", 1)[1].strip()
        elif line.startswith("TITLE:"):
            title = line.split(":", 1)[1].strip()
        elif line == "":
            i += 1
            break
        i += 1
    body = "\n".join(lines[i:]).strip()
    if not access or not title:
        raise ValueError("missing ACCESS or TITLE header")
    return access, title, body


CANARY_PATTERN = re.compile(r"CANARY-[A-Z0-9-]+")


def collect_canaries(docs_dir=None):
    """{canary: [{"title", "access"}, ...]} for every canary in the docs,
    parsed with the same ACCESS/TITLE headers ingest() uses. A canary can
    appear in more than one document; a document whose headers can't be
    parsed is recorded with access None (nobody is authorised for it)."""
    docs_dir = Path(docs_dir or DEFAULT_DOCS_DIR)
    registry = {}
    for path in sorted(docs_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            access, title, _ = _parse_doc(path)
        except ValueError:
            access, title = None, path.stem
        for canary in sorted(set(CANARY_PATTERN.findall(text))):
            registry.setdefault(canary, []).append({"title": title, "access": access})
    return registry


def _chunk(text, size=CHUNK_SIZE):
    chunks = []
    for start in range(0, len(text), size):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
    return chunks


def get_client(db_path=None):
    path = str(db_path or os.environ.get("LEAKGUARD_CHROMA_PATH", DEFAULT_DB_PATH))
    return chromadb.PersistentClient(path=path)


def ingest(docs_dir=None, db_path=None):
    docs_dir = Path(docs_dir or DEFAULT_DOCS_DIR)
    client = get_client(db_path)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(COLLECTION_NAME)

    doc_id = 0
    for path in sorted(docs_dir.glob("*.txt")):
        try:
            access, title, body = _parse_doc(path)
        except ValueError as e:
            print(f"WARNING: skipping {path.name}: {e}")
            continue
        for chunk in _chunk(body):
            if detect_injection(chunk):
                print(f"WARNING: possible prompt injection detected in {path.name}; chunk not stored.")
                continue
            collection.add(
                ids=[f"{path.stem}-{doc_id}"],
                documents=[chunk],
                metadatas=[{"access": access, "title": title}],
            )
            doc_id += 1

    return collection


if __name__ == "__main__":
    ingest()
    print("Ingestion complete.")
