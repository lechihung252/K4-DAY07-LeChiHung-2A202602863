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

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Store là một `list[dict]` trong bộ nhớ; `_make_record` chuẩn hoá mỗi `Document` thành record `{id, content, metadata, embedding}` và gọi `embedding_fn` **một lần lúc thêm** (không embed lại khi search), đồng thời `setdefault("doc_id", doc.id)` vào metadata để `delete_document` luôn có khoá để xoá. `search` chỉ là `_search_records` trên toàn bộ store: embed câu hỏi, tính `_dot` với từng record, sort giảm dần theo score và cắt `top_k`. Dùng dot product vì mọi embedder trong lab (mock, local, OpenAI, Gemini) đều trả về vector đã chuẩn hoá, nên dot ≡ cosine.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> **Lọc trước, tìm sau**: lọc metadata bằng `all(record["metadata"].get(k) == v ...)` để thu hẹp ứng viên, rồi mới đưa danh sách đó vào `_search_records`. Cách này đảm bảo `top_k` kết quả trả về đều thoả filter (nếu lọc *sau* thì có thể trả về ít hơn `top_k` hoặc rỗng). `delete_document` dựng lại list bằng list comprehension bỏ mọi record có `metadata["doc_id"] == doc_id`, so sánh độ dài trước/sau để trả về `True/False`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Gọi `store.search(question, top_k)`, đánh số từng chunk thành `[1] (source: doc_id) nội dung` rồi ghép thành khối `Context:`. Prompt yêu cầu LLM **chỉ** dùng context đã đánh số, phải trích dẫn số `[n]` đã dùng, và nói "không biết" nếu context không chứa câu trả lời — để dễ kiểm tra grounding. Có hai guard trước khi gọi LLM: store rỗng và search không trả kết quả thì trả về thông báo thay vì gọi LLM với context trống.

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

**Cấu hình của tôi:** corpus `data/university/` (2 tài liệu khởi động), frontmatter YAML được parse thành metadata (`audience`, `department`, `source_url`, `retrieved_at`, `document_version`); chunk bằng `SentenceChunker(max_sentences_per_chunk=2)` → 4 chunks; embedder `LocalEmbedder`; LLM giả lập trả về chunk `[1]` để kiểm tra grounding.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Học phần tiên quyết là gì và khi nào cần kiểm tra? | `course-registration#0` — "...Một học phần có thể yêu cầu học phần tiên quyết; sinh viên cần kiểm tra điều kiện trước khi xác nhận đăng ký." | 0.611 | ✅ | Dẫn [1]: kiểm tra điều kiện tiên quyết trước khi xác nhận đăng ký |
| 2 | Bị trùng lịch học thì phải làm sao? | `course-registration#1` — "Khi gặp lỗi trùng lịch, sinh viên điều chỉnh lớp học phần trước thời hạn điều chỉnh..." | 0.244 | ✅ (nhưng score thấp, top-2 là chunk thư viện 0.235) | Dẫn [1]: điều chỉnh lớp trước hạn điều chỉnh |
| 3 | Cần mang gì khi mượn tài liệu ở thư viện? | `library-services#0` — "...Người dùng cần mang thẻ định danh hợp lệ khi sử dụng dịch vụ mượn." | 0.714 | ✅ | Dẫn [1]: mang thẻ định danh hợp lệ |
| 4 | Gửi yêu cầu ngoại lệ về đăng ký học phần ở đâu? *(filter `audience=student`)* | `course-registration#0` — đoạn giới thiệu đăng ký học phần | 0.526 | ❌ top-1 sai; chunk đúng (`#1`, "gửi qua kênh hỗ trợ học vụ chính thức") xếp thứ 2 với 0.521 | Dẫn [1] → trả lời lạc, dù context [2] có đáp án |
| 5 | Thời hạn mượn sách thư viện là bao lâu? | `library-services#0` — giới thiệu dịch vụ mượn | 0.525 | ⚠️ đúng tài liệu nhưng **corpus chưa có** thời hạn mượn | Dẫn [1] → không có con số; LLM thật nên trả lời "không biết" |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 4 / 5 (câu 5 không tính vì đáp án chưa tồn tại trong corpus; câu 4 chunk đúng nằm ở top-2).

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *(Điền sau buổi demo nhóm.)*

### Phân tích lỗi (Bài tập 3.5)

- **Câu 4 — precision / chunk coherence:** hai chunk của cùng tài liệu chỉ chênh 0.005 điểm; chunk `#0` thắng vì chứa tiêu đề "# Đăng ký học phần" trùng với từ khoá trong câu hỏi, còn ý "yêu cầu ngoại lệ" chỉ là một câu ngắn nằm cuối chunk `#1`. Filter `audience=student` hoạt động đúng (loại chunk thư viện `audience=all`) nhưng không giúp phân biệt trong nội bộ một tài liệu. **Cải thiện:** chunk theo mục/heading để mỗi quy định là một chunk riêng, hoặc bỏ dòng tiêu đề ra khỏi nội dung embed và đưa vào metadata `title`.
- **Câu 2 — score tuyệt đối thấp (0.244):** câu hỏi dùng khẩu ngữ ("bị trùng lịch học thì phải làm sao") còn văn bản dùng ngôn ngữ hành chính ("lỗi trùng lịch... điều chỉnh lớp học phần"); mô hình MiniLM đa ngữ chỉ bắt được phần nào. Nếu đặt ngưỡng score (ví dụ 0.3) để lọc nhiễu thì câu này sẽ bị loại oan. **Cải thiện:** dùng embedder mạnh hơn cho tiếng Việt hoặc bổ sung câu hỏi mẫu (FAQ-style) vào đầu mỗi chunk.
- **Câu 5 — grounding:** retrieval trả về đúng tài liệu nhưng tài liệu chưa có thông tin; đây là lỗi *dữ liệu* chứ không phải lỗi thuật toán — nhắc nhóm phải bổ sung quy định mượn/gia hạn từ nguồn chính thức trước khi chốt gold answer, và prompt của agent phải bắt LLM nói "không biết" thay vì bịa.

--

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 9 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 8 / 10 |
| **Tổng phần cá nhân** | **57 / 60** |
