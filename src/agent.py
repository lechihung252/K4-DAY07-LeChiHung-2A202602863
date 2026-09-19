from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "No documents in the knowledge base yet — cannot answer."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "No relevant context found for this question."

        context_lines = []
        for i, r in enumerate(results, start=1):
            source = r["metadata"].get("doc_id", r["id"])
            context_lines.append(f"[{i}] (source: {source}) {r['content']}")
        context = "\n".join(context_lines)

        prompt = (
            "Answer the question using ONLY the numbered context below. "
            "Cite the context number(s) you used, e.g. [1]. "
            "If the context does not contain the answer, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
