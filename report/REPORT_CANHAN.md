# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Lê Chí Hùng
**Nhóm:** G35
**Ngày:** 19/9

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector embedding chỉ gần như cùng một hướng trong không gian ngữ nghĩa, tức là mô hình "hiểu" hai đoạn văn bản đang nói về cùng một ý, dù từ ngữ có thể khác nhau. Giá trị tiến về 1 là rất giống, về 0 là không liên quan, âm là trái nghĩa/đối lập (hiếm gặp với text embedding thực tế).

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên đăng ký học phần trên cổng học vụ."
- Câu B: "Người học ghi danh môn học qua hệ thống đào tạo."
- Tại sao tương đồng: cùng mô tả một hành động (đăng ký môn học qua hệ thống của trường); các cặp từ *sinh viên/người học*, *học phần/môn học*, *cổng học vụ/hệ thống đào tạo* là từ đồng nghĩa nên vector gần nhau dù không trùng từ nào.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Thư viện cho mượn sách tối đa 14 ngày."
- Câu B: "Giá vé xe buýt nội thành là 7.000 đồng."
- Tại sao khác: hai chủ đề hoàn toàn khác (dịch vụ thư viện vs. giao thông công cộng), không có thực thể hay hành động chung nào; điểm chung duy nhất là cùng có một con số.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine chỉ so sánh *hướng* của vector, bỏ qua *độ lớn*, nên một đoạn văn dài và một câu ngắn cùng chủ đề vẫn được coi là giống nhau; Euclid bị ảnh hưởng bởi độ lớn nên văn bản dài dễ bị "đẩy xa" mọi thứ. Ngoài ra, các mô hình embedding thường chuẩn hoá vector về độ dài 1, khi đó cosine chính là tích vô hướng — rẻ để tính và xếp hạng.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* mỗi chunk mới chỉ tiến thêm `chunk_size − overlap = 500 − 50 = 450` ký tự. Số chunk = ⌈(10 000 − 50) / 450⌉ = ⌈9 950 / 450⌉ = ⌈22,11⌉.
> *Đáp án:* **23 chunks** (22 chunk đầy 500 ký tự + 1 chunk cuối ngắn).

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Bước tiến giảm còn 400 → ⌈9 900 / 400⌉ = ⌈24,75⌉ = **25 chunks**, tức nhiều hơn 2 chunk và tốn thêm chi phí embed/lưu trữ. Ta chấp nhận điều đó để một câu hay một ý bị cắt ở ranh giới chunk vẫn xuất hiện trọn vẹn trong ít nhất một chunk, giúp truy xuất không bỏ sót ngữ cảnh ở mép chunk.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi tách câu bằng regex lookbehind `(?<=[.!?])\s+|(?<=\.)\n`: cắt tại khoảng trắng đứng *sau* dấu `.`, `!`, `?` (lookbehind nên dấu câu được giữ lại ở cuối câu, không bị mất). Sau khi `strip()` và loại các mảnh rỗng, tôi gom từng nhóm `max_sentences_per_chunk` câu bằng slicing theo bước nhảy rồi nối lại bằng dấu cách. Edge case: text rỗng trả về `[]`; `max_sentences_per_chunk` được ép tối thiểu là 1 trong `__init__`; câu cuối không có dấu chấm vẫn được giữ vì regex chỉ cắt chứ không lọc.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> `_split(text, separators)` là hàm đệ quy. **Base case**: text rỗng → `[]`; text đã ≤ `chunk_size` → trả nguyên `[text]`; hết separator (hoặc separator là `""`) → cắt cứng theo `chunk_size`. **Bước đệ quy**: tách text bằng separator đầu tiên (`\n\n` → `\n` → `. ` → ` `), mảnh nào vẫn quá dài thì gọi lại `_split` với danh sách separator còn lại. Cuối cùng có bước *merge tham lam*: gộp các mảnh liên tiếp lại (nối bằng chính separator đó) miễn là tổng vẫn ≤ `chunk_size`, để tránh sinh ra hàng loạt chunk quá nhỏ chỉ vì text có nhiều xuống dòng.

**`SectionChunker` (chiến lược tuỳ chỉnh cho Giai đoạn 2)** — hướng tiếp cận:
> Corpus là quy chế đào tạo có cấu trúc `# Điều` › `## Khoản` › điểm `a) b) c)`, nên tôi viết thêm chunker cắt đúng theo tiêu đề Markdown thay vì theo ký tự. `_split_sections` duyệt mọi heading bằng regex `^(#{1,6})\s+(.*)$` và giữ một "vệt" tiêu đề theo cấp để dựng breadcrumb `"Điều X › Khoản Y"`; `_fit` tách Khoản dài hơn `max_chars` tại ranh giới điểm `a)…` (regex `\n(?=[a-zđ]\)\s)`) nhưng lặp lại câu dẫn cho từng mảnh để mảnh nào cũng đọc được độc lập; bước cuối gộp Khoản ngắn hơn `min_chars` vào chunk liền trước. Điểm thiết kế quan trọng nhất rút ra từ ablation: breadcrumb **không** ghép vào nội dung được embed (làm mọi chunk cùng Điều giống nhau → loãng) mà trả ra qua `chunk_with_sections()` để ingest vào `metadata["section"]`; `chunk()` vẫn giữ giao diện `list[str]` như các chunker khác.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Store là một `list[dict]` trong bộ nhớ; `_make_record` chuẩn hoá mỗi `Document` thành record `{id, content, metadata, embedding}` và gọi `embedding_fn` **một lần lúc thêm** (không embed lại khi search), đồng thời `setdefault("doc_id", doc.id)` vào metadata để `delete_document` luôn có khoá để xoá. `search` chỉ là `_search_records` trên toàn bộ store: embed câu hỏi, tính `_dot` với từng record, sort giảm dần theo score và cắt `top_k`. Dùng dot product vì mọi embedder trong lab (mock, local, OpenAI, Gemini) đều trả về vector đã chuẩn hoá, nên dot ≡ cosine.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> **Lọc trước, tìm sau**: lọc metadata bằng `all(record["metadata"].get(k) == v ...)` để thu hẹp ứng viên, rồi mới đưa danh sách đó vào `_search_records`. Cách này đảm bảo `top_k` kết quả trả về đều thoả filter (nếu lọc *sau* thì có thể trả về ít hơn `top_k` hoặc rỗng). `delete_document` dựng lại list bằng list comprehension bỏ mọi record có `metadata["doc_id"] == doc_id`, so sánh độ dài trước/sau để trả về `True/False`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Gọi `store.search(question, top_k)`, đánh số từng chunk thành `[1] (source: doc_id) nội dung` rồi ghép thành khối `Context:`. Prompt yêu cầu LLM **chỉ** dùng context đã đánh số, phải trích dẫn số `[n]` đã dùng, và nói "không biết" nếu context không chứa câu trả lời — để dễ kiểm tra grounding. Có hai guard trước khi gọi LLM: store rỗng và search không trả kết quả thì trả về thông báo thay vì gọi LLM với context trống. Để phục vụ câu hỏi bắt buộc lọc `audience` của L3A, tôi thêm tham số tuỳ chọn `metadata_filter=None`: có filter thì đi qua `search_with_filter`, không thì giữ nguyên `search` (tests cũ không đổi). Nhãn nguồn trong context cũng ghép thêm `metadata["section"]` nếu có, nên LLM trích được "Điều 19 › Khoản 3" thay vì chỉ `doc_id`.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================== 42 passed in 0.07s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Embedder dùng để đo: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (`LocalEmbedder`, 384 chiều), gọi qua `compute_similarity()` của tôi. Dự đoán được ghi **trước** khi chạy.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên đăng ký học phần trên cổng học vụ. | Người học ghi danh môn học qua hệ thống đào tạo. | cao | 0.601 | ✅ |
| 2 | Thư viện cho mượn sách tối đa 14 ngày. | Giá vé xe buýt nội thành là 7.000 đồng. | thấp | 0.103 | ✅ |
| 3 | How do I renew a library book? | Làm sao để gia hạn sách thư viện? | cao | 0.731 | ✅ |
| 4 | Học phí học kỳ này bao nhiêu? | Học kỳ này học bao nhiêu tín chỉ? | thấp | 0.784 | ❌ |
| 5 | Sinh viên bị cấm thi nếu vắng quá 20% số tiết. | Sinh viên được thưởng nếu đi học đầy đủ. | thấp | 0.144 | ✅ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Cặp 4 bất ngờ nhất: hai câu hỏi về **học phí** và **tín chỉ** là hai thứ hoàn toàn khác nhau nhưng lại đạt 0.784 — cao hơn cả cặp paraphrase số 1. Lý do là embedding nắm *chủ đề và khung câu* (câu hỏi "bao nhiêu" về học kỳ, giọng sinh viên) mạnh hơn là *thực thể cụ thể* bị hỏi. Ngược lại, cặp 3 (Anh–Việt) đạt 0.731 cho thấy mô hình đa ngữ thật sự ánh xạ ý nghĩa chứ không phải từ mặt chữ. Bài học cho RAG: với corpus quy định đại học, các câu hỏi có cấu trúc giống nhau (học phí / tín chỉ / học bổng) rất dễ bị lẫn, nên cần metadata (`category`) hoặc chunk theo mục để phân biệt, không thể chỉ dựa vào cosine.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

**Cấu hình của tôi:** corpus nhóm `data/quy-dinh-dao-tao/` (10 tài liệu, 19 368 ký tự), frontmatter YAML parse thành metadata (`audience`, `category`, `article`, `source_url`, `retrieved_at`, `document_version`, …); chunk bằng **`SectionChunker(max_chars=900, min_chars=250)` v2** (chiến lược cá nhân — cắt theo Điều/Khoản, breadcrumb đưa vào `metadata["section"]`) → **36 chunks**; embedder `LocalEmbedder` (`paraphrase-multilingual-MiniLM-L12-v2`); `top_k=3`; LLM thật **OpenAI `gpt-4o-mini`** (temperature 0, system prompt yêu cầu trả lời tiếng Việt chỉ dựa vào context và trích số context). Câu 5 chạy qua `agent.answer(q, metadata_filter={"audience": "student"})`. Lệnh tái lập: `python scripts/compare_strategies.py --llm` (cần `OPENAI_API_KEY` trong `.env`; cột `hung_section`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên được đăng ký tối đa bao nhiêu TC trong học kỳ hè? | `dieu-10::c3` — Điều 10 › Khoản 2. Số lượng TC đăng ký ("…tối đa 24 TC và tối thiểu 12 TC trong học kỳ chính… tối đa 8 TC trong học kỳ hè") | 0.716 | ✅ top-1 đúng; top-2 Khoản 4 (mở lớp), top-3 Điều 19 K2 (hạn chế khối lượng) đều cùng chủ đề TC | "Sinh viên được đăng ký tối đa 8 tín chỉ trong học kỳ hè [1]." — **đúng** (2/2) |
| 2 | Khi nào sinh viên bị buộc thôi học? | `dieu-16::c3` — Điều 16 › Khoản 4 ("Khi chế độ nghỉ tạm thời có hiệu lực thì các học phần đã đăng ký… bị hủy") | 0.716 | ❌ top-1 sai; chunk đúng `dieu-19-20::c2` (Điều 19 › Khoản 3. Buộc thôi học) ở **top-2** với 0.649 | "…khi bị cảnh báo học tập mức 3 lần thứ hai liên tiếp hoặc học chậm tiến độ quá thời gian cho phép… [2]" — **đúng**, LLM tự bỏ qua [1] và trích [2]; chấm 1/2 vì chunk đúng không ở top-1 |
| 3 | Nghỉ học tạm thời vì lý do cá nhân được nghỉ tối đa bao lâu? | `dieu-16::c2` — Điều 16 › Khoản 2 (mảnh chứa điểm d: "…học ít nhất một học kỳ… tối đa 04 học kỳ chính… tính vào thời gian học chậm tiến độ") | 0.769 | ✅ top-1 đúng; top-2 là mảnh đầu của cùng Khoản 2 (0.761) | "…tối đa 04 học kỳ chính, và thời gian nghỉ này sẽ được tính vào thời gian học chậm tiến độ [1]." — **đúng** (2/2) |
| 4 | Điều kiện để được xét công nhận tốt nghiệp là gì? | `dieu-14-15::c2` — Điều 14 › Khoản 3. Điều kiện xét công nhận tốt nghiệp (trọn 4 điểm a–d trong một chunk) | 0.759 | ✅ top-1 đúng và đủ 4 điều kiện | Liệt kê đủ 4 điều kiện a)–d), nguyên văn kể cả "Giáo dục thể chất và Giáo dục quốc phòng-an ninh" [1] — **đúng** (2/2) |
| 5 | Điểm ĐATN tính từ điểm quá trình và điểm cuối kỳ theo trọng số nào? *(filter `audience=student`)* | `dieu-13-dieu-kien::c0` — Điều 13 (bản student, gộp mở đầu + Khoản 1 + Khoản 2a: "trọng số 0,5 … và 0,5 …") | 0.528 | ✅ top-1 đúng nhờ filter — **không lọc**, top-1 là `dieu-13-cham-diem` (`audience=faculty`, 0.62) | "…trọng số 0,5 đối với điểm quá trình và trọng số 0,5 đối với điểm cuối kỳ [1]." — **đúng** (2/2) |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5 / 5** (4 câu ở top-1, câu 2 ở top-2). Agent trả lời đúng **5 / 5**, luôn trích đúng số context chứa đáp án. Điểm theo `docs/SCORING.md`: 2 + 1 + 2 + 2 + 2 = **9 / 10** (câu 2 mất 1 điểm vì chunk đúng ở top-2, dù câu trả lời cuối cùng đúng).

**So với hai baseline trên cùng 5 câu** (`FixedSizeChunker(500/50)`: 8, `SentenceChunker(3)`: 8) và chiến lược của Hưởng (`RecursiveChunker` tinh chỉnh: 8): `SectionChunker` hơn đúng 1 điểm ở câu 3 — nhờ giữ trọn Khoản 2 Điều 16 (kèm câu dẫn) trong một chunk thay vì để mảnh chứa điểm d) đứng riêng. Chi tiết ablation ở `REPORT_NHOM.md` mục 2.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *(Điền sau buổi demo nhóm.)*

### Phân tích lỗi (Bài tập 3.5)

- **Câu 2 — precision / giới hạn embedder:** cả 4 chiến lược nhóm thử đều xếp các chunk của Điều 16 (nghỉ học tạm thời, *tự nguyện* thôi học) trên Khoản 3 Điều 19 (*buộc* thôi học). MiniLM đa ngữ bám vào cụm "thôi học"/"nghỉ học" và không phân biệt được sắc thái "buộc" ↔ "tự nguyện"; đây là lỗi *ngữ nghĩa của embedder* chứ không phải lỗi cắt chunk (chunk Điều 19 K3 đã là một khối trọn vẹn, có tiêu đề "Buộc thôi học" trong `section`). Với `gpt-4o-mini`, agent đọc cả 3 context và tự chọn [2] nên câu trả lời cuối cùng vẫn đúng — nhưng đó là LLM "cứu" retrieval, không nên trông chờ: nếu chunk đúng rớt khỏi top-3 (như với hai baseline Fixed/Sentence, đều miss câu này) thì LLM không có gì để cứu. **Cải thiện:** (1) lọc `metadata_filter={"category": "academic-warning"}` khi câu hỏi có từ khoá "buộc thôi học/cảnh báo" — đã có sẵn trường này; (2) hybrid search: cộng điểm BM25 để từ "buộc" có trọng số; (3) đưa `section` vào nội dung embed *chỉ với phần tên Khoản* (không kèm tên Điều) — ablation cho thấy nhồi cả breadcrumb dài làm loãng embedding.
- **Câu 5 — metadata utility:** không lọc `audience`, top-1 là tài liệu hướng dẫn chấm điểm cho giảng viên (0.62 > 0.53) vì nó lặp lại nhiều lần các cụm "điểm quá trình", "điểm cuối kỳ", "hội đồng". Score tuyệt đối của chunk đúng chỉ 0.528 — thấp nhất trong 5 câu — vì câu hỏi dùng viết tắt "ĐATN" còn chunk student rất ngắn (491 ký tự). Filter cứu được câu này nhưng cũng cho thấy **filter là điều kiện cần chứ không đủ**: nếu corpus có thêm tài liệu student nói về "điểm quá trình" (ví dụ Điều 12 đánh giá học phần) thì chunk đúng vẫn có thể rớt. **Cải thiện:** thêm trường `keywords`/`aliases` ("ĐATN" ↔ "đồ án tốt nghiệp") vào metadata hoặc mở rộng từ viết tắt trong câu hỏi trước khi embed.
- **Câu 3 — chunk coherence (điểm mạnh nhưng có giá):** Khoản 2 Điều 16 dài ~1 000 ký tự nên `SectionChunker` phải tách thành 2 mảnh theo điểm a)–c) / d)–đ), mỗi mảnh vẫn giữ câu dẫn. Nhờ vậy mảnh chứa d) lên top-1; nhưng hai mảnh cùng Khoản chiếm top-1 và top-2 (0.769 / 0.761), đẩy các Khoản khác ra khỏi top-3 → nếu câu hỏi cần thông tin từ 2 Khoản khác nhau thì `top_k=3` sẽ thiếu. **Cải thiện:** khử trùng lặp theo `section` khi lấy top-k (mỗi Khoản chỉ giữ mảnh điểm cao nhất) hoặc tăng `top_k` lên 5 cho câu hỏi tổng hợp.

--

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 9 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 9 / 10 |
| **Tổng phần cá nhân** | **58 / 60** |
