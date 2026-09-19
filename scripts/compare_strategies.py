"""So sánh chiến lược chunking của các thành viên nhóm G35 trên corpus quy chế đào tạo.

Chạy:  python scripts/compare_strategies.py            (LocalEmbedder nếu cài, ngược lại mock)
       python scripts/compare_strategies.py --mock

In ra các bảng Markdown để dán vào report/REPORT_NHOM.md mục 2.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import (  # noqa: E402
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    RecursiveChunker,
    SectionChunker,
    SentenceChunker,
    _mock_embed,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "quy-dinh-dao-tao"
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)

# --- Chiến lược của từng thành viên ---------------------------------------
STRATEGIES = {
    "baseline_fixed(500/50)": FixedSizeChunker(chunk_size=500, overlap=50),
    "baseline_sentence(3)": SentenceChunker(max_sentences_per_chunk=3),
    # Thành viên 2 — Hưởng: RecursiveChunker tinh chỉnh, thêm "\n## " để ưu tiên cắt tại Khoản
    "huong_recursive(600)": RecursiveChunker(
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""], chunk_size=600
    ),
    # Thành viên 1 — Hùng: SectionChunker custom, cắt theo Điều/Khoản + breadcrumb
    "hung_section(900/250)": SectionChunker(max_chars=900, min_chars=250),  # breadcrumb → metadata["section"]
}

# --- 5 câu hỏi đánh giá của nhóm (xem REPORT_NHOM.md mục 3) -----------------
# (query, gold doc_id, chuỗi phải xuất hiện trong chunk đúng, metadata_filter)
QUERIES = [
    ("Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong học kỳ hè?",
     "dieu-10-dang-ky-hoc-tap", "tối đa 8 TC", None),
    ("Khi nào sinh viên bị buộc thôi học?",
     "dieu-19-20-canh-bao-hoc-tap-va-buoc-thoi-hoc", "mức 3 lần thứ hai", None),
    ("Nghỉ học tạm thời vì lý do cá nhân thì được nghỉ tối đa bao lâu?",
     "dieu-16-nghi-hoc-tam-thoi-va-thoi-hoc", "04 học kỳ chính", None),
    ("Điều kiện để được xét công nhận tốt nghiệp là gì?",
     "dieu-14-15-dang-ky-tot-nghiep-va-hang-tot-nghiep", "chuẩn ngoại ngữ", None),
    ("Điểm ĐATN được tính từ điểm quá trình và điểm cuối kỳ theo trọng số nào?",
     "dieu-13-dieu-kien-lam-do-an-tot-nghiep", "trọng số 0,5", {"audience": "student"}),
]


def load_corpus() -> list[tuple[str, dict, str]]:
    docs = []
    for path in sorted(DATA_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta: dict = {}
        m = FRONTMATTER_RE.match(raw)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"')
        body = raw[m.end():].strip() if m else raw
        docs.append((meta.get("doc_id", path.stem), meta, body))
    return docs


def get_embedder(use_mock: bool):
    if not use_mock:
        try:
            from src import LocalEmbedder
            emb = LocalEmbedder()
            return emb, emb._backend_name
        except Exception as exc:  # noqa: BLE001
            print(f"(LocalEmbedder không khả dụng: {exc} → dùng mock)")
    return _mock_embed, "mock"


def build_store(chunker, corpus, embed_fn) -> EmbeddingStore:
    store = EmbeddingStore(embedding_fn=embed_fn)
    docs = []
    for doc_id, meta, body in corpus:
        if hasattr(chunker, "chunk_with_sections"):
            pairs = chunker.chunk_with_sections(body)
        else:
            pairs = [("", c) for c in chunker.chunk(body)]
        for i, (section, chunk) in enumerate(pairs):
            docs.append(Document(id=f"{doc_id}::c{i}", content=chunk,
                                 metadata={**meta, "doc_id": doc_id, "chunk_index": i, "section": section}))
    store.add_documents(docs)
    return store


def score_hits(results, gold_doc, gold_text) -> tuple[int, str]:
    """2 = top-1 đúng chunk; 1 = đúng chunk ở top-2/3; 0 = không có trong top-3."""
    for rank, r in enumerate(results, 1):
        if r["metadata"]["doc_id"] == gold_doc and gold_text in r["content"]:
            return (2 if rank == 1 else 1), f"top-{rank}"
    return 0, "miss"


def main() -> None:
    use_mock = "--mock" in sys.argv
    corpus = load_corpus()
    embed_fn, backend = get_embedder(use_mock)
    print(f"Embedder: {backend}\nCorpus: {len(corpus)} tài liệu, {sum(len(b) for _, _, b in corpus)} ký tự\n")

    # 1) Baseline comparator trên 3 tài liệu
    print("## Baseline — ChunkingStrategyComparator().compare(text, chunk_size=200)\n")
    print("| Tài liệu | Chiến lược | Số chunk | Độ dài TB | Chunk bị cắt giữa Khoản? |")
    print("|---|---|---|---|---|")
    comparator = ChunkingStrategyComparator()
    for doc_id, _, body in corpus:
        if doc_id not in {"dieu-10-dang-ky-hoc-tap", "dieu-16-nghi-hoc-tam-thoi-va-thoi-hoc",
                          "dieu-19-20-canh-bao-hoc-tap-va-buoc-thoi-hoc"}:
            continue
        res = comparator.compare(body, chunk_size=200)
        n_sections = len(re.findall(r"^## ", body, re.MULTILINE))
        for name, r in res.items():
            # chunk "cắt ngang" = chunk không bắt đầu bằng tiêu đề và không bắt đầu bằng chữ hoa/ký hiệu điểm
            broken = sum(1 for c in r["chunks"] if not re.match(r"^(#|[A-ZĐ]|[a-zđ]\)|\d)", c.strip()))
            print(f"| {doc_id} ({n_sections} Khoản) | {name} | {r['count']} | {r['avg_length']:.0f} | {broken}/{r['count']} |")
    print()

    # 2) Thống kê chunk toàn corpus theo chiến lược
    print("## Thống kê chunk toàn corpus (10 tài liệu)\n")
    print("| Chiến lược | Tổng chunk | Độ dài TB | Min | Max | Chunk bắt đầu đúng ranh giới Khoản/điểm |")
    print("|---|---|---|---|---|---|")
    stores = {}
    for name, chunker in STRATEGIES.items():
        all_chunks = [c for _, _, body in corpus for c in chunker.chunk(body)]
        lens = [len(c) for c in all_chunks]
        aligned = sum(1 for c in all_chunks if re.match(r"^(#|Điều|[A-ZĐ]|[a-zđ]\))", c.strip()))
        print(f"| {name} | {len(all_chunks)} | {sum(lens)/len(lens):.0f} | {min(lens)} | {max(lens)} | {aligned}/{len(all_chunks)} ({100*aligned/len(all_chunks):.0f}%) |")
        stores[name] = build_store(chunker, corpus, embed_fn)
    print()

    # 3) Benchmark 5 câu hỏi
    print("## Kết quả 5 câu hỏi (top-3, điểm ước lượng 2/1/0 theo vị trí chunk đúng)\n")
    totals = {name: 0 for name in STRATEGIES}
    header = "| # | Câu hỏi | " + " | ".join(STRATEGIES) + " |"
    print(header)
    print("|" + "---|" * (2 + len(STRATEGIES)))
    details = []
    for qi, (q, gold_doc, gold_text, flt) in enumerate(QUERIES, 1):
        row = []
        for name, store in stores.items():
            results = store.search_with_filter(q, top_k=3, metadata_filter=flt) if flt else store.search(q, top_k=3)
            s, where = score_hits(results, gold_doc, gold_text)
            totals[name] += s
            row.append(f"{s} ({where})")
            details.append((qi, name, results))
        flag = " *(filter audience=student)*" if flt else ""
        print(f"| {qi} | {q}{flag} | " + " | ".join(row) + " |")
    print("| | **Tổng /10** | " + " | ".join(f"**{totals[n]}**" for n in STRATEGIES) + " |")
    print()

    print("## Chi tiết top-3\n")
    for qi, name, results in details:
        print(f"**Q{qi} — {name}**")
        for r in results:
            snippet = r["content"].replace("\n", " ")[:90]
            print(f"- {r['score']:.3f} `{r['id']}` — {snippet}…")
        print()


if __name__ == "__main__":
    main()
