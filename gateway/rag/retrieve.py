import os
import re

from gateway.policy import RAG
from gateway.rag.ingest import COLLECTION_NAME, DEFAULT_DB_PATH, get_client

ROLE_ACCESS = {
    "intern": ["public"],
    "manager": ["public", "internal"],
    "hr": ["public", "internal", "hr"],
}
ALL_ACCESS_LEVELS = ["public", "internal", "hr"]

# Two-tier relevance (distances are Chroma's default squared L2):
#  - at or below RELEVANCE_THRESHOLD a chunk is relevant on meaning alone;
#  - up to KEYWORD_RELEVANCE_THRESHOLD it is relevant only if it also shares a
#    distinctive word with the query (one that appears in few documents, so
#    "policy" doesn't count but "casual" or "insured" does);
#  - above that it is dropped.
# Embedding distance alone can't separate these: "How many casual leaves do
# I get?" matches Leave Policy at ~1.50, while the unrelated "What is the
# sum insured under this policy?" matches Travel Reimbursement at ~1.32.
RELEVANCE_THRESHOLD = RAG.get("relevance_threshold", 1.25)
KEYWORD_RELEVANCE_THRESHOLD = RAG.get("keyword_relevance_threshold", 1.65)
KEYWORD_MAX_DOC_FRACTION = RAG.get("keyword_max_doc_fraction", 0.2)

_STOPWORDS = set("""
a an the and or but if then than so to of in on at by for with from into onto about
over under up down out off is are was were be been being am do does did done have has
had having i me my mine we us our ours you your yours he him his she her hers it its
they them their what which who whom whose when where why how this that these those
there here can could will would shall should may might must not no yes all any each
every some such many much more most other own same also just only very too get got
give gets tell show please want need know like one two per new use used using make
""".split())

_SUFFIXES = ("ations", "ation", "ances", "ance", "ences", "ence", "ings", "ing", "ies",
             "ied", "ers", "er", "es", "ed", "s", "y", "e")


def _stem(word):
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def _terms(text):
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) >= 3 and w not in _STOPWORDS}


_TERM_STATS = {}  # (db path, chunk count) -> ({term: number of documents}, number of documents)


def _doc_frequencies(collection, cache_key):
    if cache_key not in _TERM_STATS:
        everything = collection.get(include=["documents", "metadatas"])
        per_doc = {}
        for doc, meta in zip(everything.get("documents") or [],
                             everything.get("metadatas") or []):
            per_doc.setdefault((meta or {}).get("title"), set()).update(_terms(doc))
        df = {}
        for terms in per_doc.values():
            for t in terms:
                df[t] = df.get(t, 0) + 1
        _TERM_STATS.clear()
        _TERM_STATS[cache_key] = (df, max(len(per_doc), 1))
    return _TERM_STATS[cache_key]


def _shares_distinctive_term(query, chunk_text, df, n_docs):
    max_df = max(1, int(KEYWORD_MAX_DOC_FRACTION * n_docs))
    shared = _terms(query) & _terms(chunk_text)
    return any(df.get(t, 0) <= max_df for t in shared)

# The restricted_doc check is stricter still: the best restricted-access
# match must itself be close (below restricted_doc_threshold) AND clearly
# closer than the best match the role is actually allowed to see (by at
# least restricted_doc_margin). Both conditions must hold, so a merely
# "less dissimilar" restricted document never triggers an approval request
# for an unrelated prompt.
RESTRICTED_DOC_THRESHOLD = RAG.get("restricted_doc_threshold", 1.3)
RESTRICTED_DOC_MARGIN = RAG.get("restricted_doc_margin", 0.3)


def _db_key(db_path=None):
    return str(db_path or os.environ.get("LEAKGUARD_CHROMA_PATH", DEFAULT_DB_PATH))


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
    df, n_docs = _doc_frequencies(collection, (_db_key(db_path), collection.count()))
    for doc, meta, dist in zip(docs, metas, dists):
        # Defense in depth on top of the where-filter: a chunk this role
        # can't see (or with missing/unknown access metadata) never passes.
        if (meta or {}).get("access") not in allowed:
            continue
        if dist > KEYWORD_RELEVANCE_THRESHOLD:
            continue
        if dist > RELEVANCE_THRESHOLD and not _shares_distinctive_term(query, doc, df, n_docs):
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
