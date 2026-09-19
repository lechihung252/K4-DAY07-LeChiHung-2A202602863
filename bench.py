"""bench.py — công cụ đo retrieval cá nhân (Lê Chí Hùng, nhóm G35).

Chạy:  python bench.py            → nạp corpus, chunk bằng chiến lược của tôi, in top-3 cho 5 câu
       python bench.py --llm      → thêm câu trả lời của KnowledgeBaseAgent qua OpenAI (cần .env)
       python bench.py --mock     → dùng MockEmbedder (chỉ để kiểm tra script chạy, số liệu là nhiễu)

Kết quả nộp bài: python bench.py --llm > ket_qua_benchmark.txt
Bốn việc theo codelab: (1) đọc .md + tách frontmatter, (2) chunk NGOÀI store → Document("file#i"),
(3) nạp EmbeddingStore, chạy 5 query qua search_with_filter, (4) in top-3 kèm score + doc_id.
So sánh nhiều chiến lược cùng lúc: scripts/compare_strategies.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # tắt cảnh báo fork của HF tokenizers
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from compare_strategies import (  # noqa: E402  (nạp .env, load_corpus, QUERIES, score_hits, make_openai_llm)
    DATA_DIR,
    QUERIES,
    load_corpus,
    make_openai_llm,
    score_hits,
)
from src import Document, EmbeddingStore, KnowledgeBaseAgent, SectionChunker, _mock_embed  # noqa: E402

# ---- Dòng duy nhất mỗi thành viên đổi: chiến lược chunking của mình ----------------
CHUNKER = SectionChunker(max_chars=900, min_chars=250)   # Hùng: cắt theo Điều/Khoản, breadcrumb → metadata["section"]
# Hưởng: RecursiveChunker(separators=["\n## ", "\n\n", "\n", ". ", " ", ""], chunk_size=600)
# ------------------------------------------------------------------------------------

TOP_K = 3


def chunk_pairs(text: str) -> list[tuple[str, str]]:
    """Trả về [(section, chunk)]; chunker không có chunk_with_sections thì section rỗng."""
    if hasattr(CHUNKER, "chunk_with_sections"):
        return CHUNKER.chunk_with_sections(text)
    return [("", c) for c in CHUNKER.chunk(text)]


def main() -> None:
    use_mock = "--mock" in sys.argv
    if use_mock:
        embed_fn, backend = _mock_embed, "mock (số liệu chỉ để kiểm tra script)"
    else:
        from src import LocalEmbedder
        embedder = LocalEmbedder()
        embed_fn, backend = embedder, embedder._backend_name

    print(f"# ket_qua_benchmark — Lê Chí Hùng (G35) — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"corpus   : {DATA_DIR.relative_to(Path.cwd()) if DATA_DIR.is_relative_to(Path.cwd()) else DATA_DIR}")
    print(f"chunker  : {CHUNKER.__class__.__name__} {vars(CHUNKER)}")
    print(f"embedder : {backend}")
    print(f"top_k    : {TOP_K}\n")

    # 1–2. đọc file, tách frontmatter, chunk ngoài store
    corpus = load_corpus()
    docs: list[Document] = []
    for doc_id, meta, body in corpus:
        for i, (section, chunk) in enumerate(chunk_pairs(body)):
            docs.append(Document(
                id=f"{doc_id}#{i}",
                content=chunk,
                metadata={**meta, "doc_id": doc_id, "chunk_index": i, "section": section},
            ))

    # 3. nạp store
    store = EmbeddingStore(embedding_fn=embed_fn)
    store.add_documents(docs)
    print(f"Đã nạp {len(corpus)} tài liệu → {store.get_collection_size()} chunk "
          f"(TB {sum(len(d.content) for d in docs) / len(docs):.0f} ký tự/chunk)\n")

    # 4. chạy 5 query, in top-3
    total = 0
    for qi, (q, gold_doc, gold_text, flt) in enumerate(QUERIES, 1):
        results = store.search_with_filter(q, top_k=TOP_K, metadata_filter=flt)
        score, where = score_hits(results, gold_doc, gold_text)
        total += score
        print(f"Q{qi}. {q}")
        print(f"    filter : {flt or '—'}    gold: {gold_doc} ∋ \"{gold_text}\"    → {where}, {score}/2")
        for rank, r in enumerate(results, 1):
            hit = "✓" if (r["metadata"]["doc_id"] == gold_doc and gold_text in r["content"]) else " "
            section = r["metadata"].get("section") or "(không có section)"
            preview = r["content"].replace("\n", " ")[:100]
            print(f"    {hit} top-{rank}  {r['score']:.4f}  {r['id']:<52} audience={r['metadata'].get('audience')}")
            print(f"              {section}")
            print(f"              {preview}…")
        print()
    print(f"Điểm truy xuất (2 = gold ở top-1, 1 = top-2/3, 0 = miss): {total}/10\n")

    # tuỳ chọn: agent trả lời bằng LLM thật
    if "--llm" in sys.argv:
        if not os.getenv("OPENAI_API_KEY"):
            print("(--llm bỏ qua: chưa có OPENAI_API_KEY trong .env)")
            return
        llm_fn = make_openai_llm()
        agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
        print(f"## Câu trả lời của KnowledgeBaseAgent — LLM {llm_fn.model}\n")
        for qi, (q, _, gold_text, flt) in enumerate(QUERIES, 1):
            answer = agent.answer(q, top_k=TOP_K, metadata_filter=flt)
            print(f"Q{qi}. {q}{'  [filter ' + str(flt) + ']' if flt else ''}")
            print(f"    gold : {gold_text}")
            print(f"    agent: {answer}\n")


if __name__ == "__main__":
    main()
