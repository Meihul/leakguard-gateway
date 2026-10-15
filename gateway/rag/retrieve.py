from gateway.policy import RAG
from gateway.rag.ingest import COLLECTION_NAME, get_client

ROLE_ACCESS = {
    "intern": ["public"],
    "manager": ["public", "internal"],
    "hr": ["public", "internal", "hr"],
}

# Below this cosine distance a match is considered actually relevant to the
# query, rather than just the "least dissimilar" of the corpus. Chunks above
# it are dropped (so an unrelated question returns no chunks/sources at all),
# and the restricted_doc check uses the same threshold, so an unrelated
# question is never sent for approval just because some access-restricted
# document happened to be the nearest match.
RELEVANCE_THRESHOLD = RAG.get("relevance_threshold", 1.3)


def _get_collection(db_path=None):
    client = get_client(db_path)
    return client.get_or_create_collection(COLLECTION_NAME)


def retrieve(query, role, db_path=None, n_results=3):
    allowed = ROLE_ACCESS.get(role, ["public"])
    collection = _get_collection(db_path)

    chunks = []
    findings = []

    if collection.count() == 0:
        return {"chunks": chunks, "findings": findings}

    filtered = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        where={"access": {"$in": allowed}},
    )
    docs = (filtered.get("documents") or [[]])[0]
    metas = (filtered.get("metadatas") or [[]])[0]
    dists = (filtered.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        if dist > RELEVANCE_THRESHOLD:
            continue
        chunks.append({"text": doc, "title": meta.get("title"), "access": meta.get("access")})

    unfiltered = collection.query(query_texts=[query], n_results=1)
    u_metas = (unfiltered.get("metadatas") or [[]])[0]
    u_dists = (unfiltered.get("distances") or [[]])[0]
    if u_metas and u_dists and u_dists[0] <= RELEVANCE_THRESHOLD:
        best = u_metas[0]
        if best.get("access") not in allowed:
            findings.append({
                "category": "restricted_doc",
                "span": best.get("title"),
                "start": 0,
                "end": 0,
                "severity": "high",
                "confidence": 1.0,
                "detector": "rag.access_control",
            })

    return {"chunks": chunks, "findings": findings}


def get_chunks_by_title(title, db_path=None):
    """Fetch every chunk for a document by title, ignoring access control.

    Used only once a restricted document has been explicitly approved, to
    build the LLM context from the specific document that was requested.
    """
    collection = _get_collection(db_path)
    if collection.count() == 0:
        return []
    result = collection.get(where={"title": title})
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    return [{"text": d, "title": m.get("title"), "access": m.get("access")}
            for d, m in zip(docs, metas)]
