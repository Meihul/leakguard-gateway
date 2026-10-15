from gateway.policy import RAG
from gateway.rag.ingest import COLLECTION_NAME, get_client

ROLE_ACCESS = {
    "intern": ["public"],
    "manager": ["public", "internal"],
    "hr": ["public", "internal", "hr"],
}
ALL_ACCESS_LEVELS = ["public", "internal", "hr"]

# Below this cosine distance a match is considered actually relevant to the
# query, rather than just the "least dissimilar" of the corpus. Chunks above
# it are dropped, so an unrelated question returns no chunks/sources at all.
RELEVANCE_THRESHOLD = RAG.get("relevance_threshold", 1.3)

# The restricted_doc check is stricter still: the best restricted-access
# match must itself be close (below restricted_doc_threshold) AND clearly
# closer than the best match the role is actually allowed to see (by at
# least restricted_doc_margin). Both conditions must hold, so a merely
# "less dissimilar" restricted document never triggers an approval request
# for an unrelated prompt.
RESTRICTED_DOC_THRESHOLD = RAG.get("restricted_doc_threshold", 1.3)
RESTRICTED_DOC_MARGIN = RAG.get("restricted_doc_margin", 0.3)


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
    best_allowed_dist = dists[0] if dists else None
    for doc, meta, dist in zip(docs, metas, dists):
        if dist > RELEVANCE_THRESHOLD:
            continue
        chunks.append({"text": doc, "title": meta.get("title"), "access": meta.get("access")})

    restricted_levels = [lvl for lvl in ALL_ACCESS_LEVELS if lvl not in allowed]
    if restricted_levels:
        restricted = collection.query(
            query_texts=[query], n_results=1,
            where={"access": {"$in": restricted_levels}},
        )
        r_metas = (restricted.get("metadatas") or [[]])[0]
        r_dists = (restricted.get("distances") or [[]])[0]
        if r_metas and r_dists:
            best_restricted_dist = r_dists[0]
            close_enough = best_restricted_dist <= RESTRICTED_DOC_THRESHOLD
            clearly_closer = (best_allowed_dist is None
                              or (best_allowed_dist - best_restricted_dist) >= RESTRICTED_DOC_MARGIN)
            if close_enough and clearly_closer:
                findings.append({
                    "category": "restricted_doc",
                    "span": r_metas[0].get("title"),
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
