from gateway.rag.ingest import COLLECTION_NAME, get_client

ROLE_ACCESS = {
    "intern": ["public"],
    "manager": ["public", "internal"],
    "hr": ["public", "internal", "hr"],
}

# Below this cosine distance the single best match is considered actually
# relevant to the query, rather than just the "least dissimilar" of a small
# corpus; a restricted_doc finding only fires on relevant matches.
RELEVANCE_THRESHOLD = 1.3


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
    for doc, meta in zip(docs, metas):
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
