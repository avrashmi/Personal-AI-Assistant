def rerank_docs(query, docs):
    """Simple keyword-based reranking.

    Scores retrieved documents based on
    keyword overlap with the user question."""

    query_words = set(query.lower().split())

    scored_docs = []
    for doc in docs:
        text_words = set(doc.page_content.lower().split())
        score = len(query_words.intersection(text_words))
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs[:2]]