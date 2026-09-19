# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** G35
**Thành viên:** Lê Chí Hùng
**Thành viên:** Nguyễn Văn Hưởng
**Ngày:** 19/9

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** [ví dụ: Customer support FAQ, Luật Việt Nam, công thức nấu ăn, ...]

**Tại sao nhóm chọn chủ đề này?**
> *Viết 2-3 câu:*

### Danh sách tài liệu (Data Inventory)

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [ ] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [ ] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

> Mỗi thành viên thử **một chiến lược khác nhau** trên cùng bộ tài liệu; nhóm tổng hợp và so sánh ở đây.
>
> Toàn bộ số liệu dưới đây được sinh bởi `python scripts/compare_strategies.py` (embedder thật: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, corpus 10 tài liệu trong `data/quy-dinh-dao-tao/`, 19 368 ký tự nội dung sau khi bỏ frontmatter). Chạy lại script sẽ ra đúng các bảng này.

### Phân tích đường cơ sở (Baseline Analysis)

Chạy `ChunkingStrategyComparator().compare(text, chunk_size=200)` trên 3 tài liệu. Cột cuối đếm số chunk **bắt đầu giữa câu/giữa Khoản** (không mở đầu bằng tiêu đề, chữ hoa hoặc ký hiệu điểm `a)`, `b)`…):

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| Điều 10 — Đăng ký học tập (6 Khoản) | FixedSizeChunker (`fixed_size`) | 16 | 198 | ✗ 15/16 chunk cắt giữa câu |
| Điều 10 — Đăng ký học tập (6 Khoản) | SentenceChunker (`by_sentences`) | 8 | 394 | ~ 0 chunk cắt giữa câu, nhưng 1 chunk gộp cuối Khoản 1 với đầu Khoản 2 |
| Điều 10 — Đăng ký học tập (6 Khoản) | RecursiveChunker (`recursive`) | 22 | 142 | ~ 2/22 cắt giữa câu; chunk quá vụn (một điểm `a)` = một chunk) |
| Điều 16 — Nghỉ học tạm thời (5 Khoản) | FixedSizeChunker (`fixed_size`) | 12 | 188 | ✗ 11/12 |
| Điều 16 — Nghỉ học tạm thời (5 Khoản) | SentenceChunker (`by_sentences`) | 6 | 374 | ~ 0/6 |
| Điều 16 — Nghỉ học tạm thời (5 Khoản) | RecursiveChunker (`recursive`) | 16 | 140 | ~ 2/16 |
| Điều 19–20 — Cảnh báo học tập (3 Khoản) | FixedSizeChunker (`fixed_size`) | 11 | 183 | ✗ 7/11 |
| Điều 19–20 — Cảnh báo học tập (3 Khoản) | SentenceChunker (`by_sentences`) | 6 | 333 | ~ 0/6 |
| Điều 19–20 — Cảnh báo học tập (3 Khoản) | RecursiveChunker (`recursive`) | 15 | 132 | ✓ 0/15 |

**Nhận xét baseline:**
- `fixed_size` gần như luôn cắt giữa câu (ví dụ chunk bắt đầu bằng `"hè không có đợt điều chỉnh đăng ký. ## Khoản 2…"`), làm mất chủ ngữ của quy định — tệ nhất cho văn bản pháp quy.
- `by_sentences` giữ trọn câu nhưng **không biết ranh giới Khoản**: regex tách câu theo `". "` nên các điểm `a) … ; b) …` (kết thúc bằng `;`) bị dính thành một chunk dài, còn tiêu đề `## Khoản 2` lại bị dính vào cuối chunk của Khoản 1.
- `recursive` với `chunk_size=200` tôn trọng dòng trống nên ít cắt giữa câu, nhưng tạo chunk quá nhỏ (mỗi điểm `a)`, `b)` một chunk) → mất câu dẫn "được quy định như sau:" phía trên → chunk đứng một mình không hiểu đang nói về điều kiện gì.

→ Kết luận chung của nhóm: với quy chế đào tạo, **đơn vị ý nghĩa là Khoản**, nên chiến lược phải nhận biết được tiêu đề `## Khoản`. Hai thành viên đi hai hướng khác nhau để đạt điều đó.

### Chiến lược của từng thành viên

**Thành viên 1 — Lê Chí Hùng**
- **Loại chiến lược:** custom — `SectionChunker` (chia theo tiêu đề Điều/Khoản, có breadcrumb) — đáp ứng yêu cầu L3A "ít nhất một thành viên chunk theo heading/section".
- **Mô tả & lý do chọn cho chủ đề này:** Mỗi Khoản của quy chế là một quy định trọn vẹn (điều kiện + hệ quả), nên cắt đúng tại `## Khoản` giữ được ngữ cảnh pháp lý tốt nhất. Khoản dài hơn `max_chars=900` được tách tiếp theo các điểm `a)…` nhưng **giữ lại câu dẫn** cho từng mảnh; Khoản quá ngắn (< `min_chars=250`) được gộp vào chunk trước để tránh chunk chỉ có một câu. Mỗi chunk đi kèm *breadcrumb* `"<tên Điều> › <tên Khoản>"`: bản v1 ghép breadcrumb vào đầu nội dung; bản v2 (chọn) đưa breadcrumb vào `metadata["section"]` qua `chunk_with_sections()` để agent vẫn trích dẫn được "Điều X Khoản Y" mà embedding không bị loãng bởi tiền tố lặp lại (xem ablation bên dưới).
- **Code snippet:** đầy đủ tại `src/chunking.py` (`class SectionChunker`), phần lõi:
```python
class SectionChunker:
    HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
    FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
    ITEM_RE = re.compile(r"\n(?=[a-zđ]\)\s)")          # ranh giới điểm a) b) c)

    def __init__(self, max_chars=900, min_chars=250, include_breadcrumb=False): ...

    def chunk(self, text: str) -> list[str]:
        pairs = self.chunk_with_sections(text)
        return [body for _, body in pairs]                    # v2: breadcrumb đi vào metadata

    def chunk_with_sections(self, text: str) -> list[tuple[str, str]]:
        text = self.FRONTMATTER_RE.sub("", text).strip()
        pieces = []
        for breadcrumb, body in self._split_sections(text):   # ("Điều 19 … › Khoản 3. Buộc thôi học", "Buộc thôi học là …")
            for piece in self._fit(body):                     # tách theo a) b) c) nếu body > max_chars, giữ câu dẫn
                pieces.append((breadcrumb, piece))
        merged = []                                           # gộp khoản quá ngắn vào chunk liền trước
        for crumb, body in pieces:
            if merged and len(body) < self.min_chars:
                merged[-1] = (merged[-1][0], merged[-1][1] + "\n" + body)
            else:
                merged.append((crumb, body))
        return merged
```
Khi ingest: `Document(content=body, metadata={..., "section": breadcrumb})` (xem `build_store()` trong `scripts/compare_strategies.py`).

**Thành viên 2 — Nguyễn Văn Hưởng**
- **Loại chiến lược:** Recursive (tinh chỉnh) — `RecursiveChunker(separators=["\n## ", "\n\n", "\n", ". ", " ", ""], chunk_size=600)`.
- **Mô tả & lý do chọn:** Giữ nguyên thuật toán built-in nhưng **thêm `"\n## "` lên đầu danh sách separator** để chunker ưu tiên cắt tại Khoản trước khi cắt theo đoạn/câu, và nâng `chunk_size` từ 200 → 600 để một Khoản trung bình (~400–500 ký tự) nằm trọn trong một chunk thay vì bị vỡ thành từng điểm `a)`, `b)`. Ưu điểm là không cần viết code mới, không có breadcrumb nên chunk gọn; nhược điểm là khi cắt bằng separator thì tiêu đề `## ` bị mất dấu `#`, và các Khoản ngắn liên tiếp bị gộp vào nhau theo giới hạn ký tự chứ không theo nghĩa.
- **Code snippet:** không custom — chỉ đổi tham số:
```python
RecursiveChunker(separators=["\n## ", "\n\n", "\n", ". ", " ", ""], chunk_size=600)
```

### Thống kê chunk trên toàn corpus (10 tài liệu)

| Chiến lược | Tổng chunk | Độ dài TB | Min | Max | Chunk bắt đầu đúng ranh giới Khoản/điểm |
|---|---|---|---|---|---|
| Baseline `FixedSizeChunker(500, 50)` | 47 | 451 | 62 | 500 | 11/47 (23%) |
| Baseline `SentenceChunker(3)` | 49 | 393 | 106 | 780 | 49/49 (100%) |
| Hưởng — `RecursiveChunker(600, "\n## ")` | 45 | 428 | 103 | 600 | 45/45 (100%) |
| Hùng — `SectionChunker(900, 250)` v2 | 36 | 511 | 79 | 916 | 36/36 (100%) |

### So Sánh Giữa Các Thành Viên

Chạy 5 câu hỏi đánh giá của nhóm (mục 3) với `top_k=3`. Điểm ở đây **chỉ tính vị trí chunk chứa gold answer**: **2** = top-1, **1** = top-2/3, **0** = không có trong top-3; phần "agent trả lời đúng" mỗi thành viên chấm trên code cá nhân ở `REPORT_CANHAN.md`.

| # | Câu hỏi | Baseline Fixed | Baseline Sentence | Hưởng — Recursive | Hùng — Section v2 |
|---|---|---|---|---|---|
| 1 | Sinh viên được đăng ký tối đa bao nhiêu TC trong học kỳ hè? | 2 | 2 | 2 | 2 |
| 2 | Khi nào sinh viên bị buộc thôi học? | 0 (miss) | 0 (miss) | 1 (top-2) | 1 (top-2) |
| 3 | Nghỉ học tạm thời vì lý do cá nhân được nghỉ tối đa bao lâu? | 2 | 2 | 1 (top-2) | 2 |
| 4 | Điều kiện để được xét công nhận tốt nghiệp là gì? | 2 | 2 | 2 | 2 |
| 5 | Điểm ĐATN tính từ điểm quá trình và điểm cuối kỳ theo trọng số nào? *(filter `audience=student`)* | 2 | 2 | 2 | 2 |
| | **Tổng /10** | **8** | **8** | **8** | **9** |

Điểm số sát nhau vì corpus nhỏ (19k ký tự) và các câu 1, 4, 5 đều có từ khoá rất đặc trưng — mọi cách cắt đều tìm ra. Khác biệt chỉ xuất hiện ở hai câu khó: **Q2** (baseline miss hoàn toàn, hai chiến lược nhận biết Khoản đưa được đúng chunk vào top-2) và **Q3** (chunk gộp trọn Khoản 2 của Section lên top-1, còn Recursive cắt Khoản 2 thành 2 mảnh nên mảnh chứa điểm `d)` tụt xuống top-2).

| Thành viên | Chiến lược (Strategy) | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Lê Chí Hùng | `SectionChunker` v2 (custom, theo Điều/Khoản; breadcrumb trong metadata) | 9 | 100% chunk đúng ranh giới Khoản; ít chunk nhất (36) nên ít nhiễu; một Khoản liệt kê điều kiện nằm trọn một chunk (thắng Q3); `metadata["section"]` cho phép agent trích "Điều 16 Khoản 2" | Chunk dài nhất (TB 511, max 916) → tốn token prompt hơn; phụ thuộc vào việc tài liệu có heading Markdown chuẩn (`# Điều`, `## Khoản`); Q2 vẫn thua chunk "tự nguyện thôi học" |
| Nguyễn Văn Hưởng | `RecursiveChunker` tinh chỉnh (`"\n## "`, 600) | 8 | Không cần code mới; chunk vừa phải (TB 428); tôn trọng ranh giới Khoản nhờ separator `"\n## "`; Q2 ngang Section | Cắt theo giới hạn ký tự nên Khoản dài (Điều 16 Khoản 2, ~1000 ký tự) bị vỡ thành 2 mảnh theo dòng trống, mảnh sau mất câu dẫn (thua Q3); tiêu đề mất dấu `##`; không có ngữ cảnh Điều để trích nguồn |
| (baseline) | `FixedSizeChunker(500/50)` | 8 | Overlap 50 tình cờ giữ được câu chứa đáp án ở Q1, Q3, Q5 | 77% chunk cắt giữa câu (ví dụ chunk mở đầu `"hè không có đợt điều chỉnh đăng ký. ## Khoản 2…"`); Q2 miss; điểm similarity thấp nhất ở Q5 (0.487) |
| (baseline) | `SentenceChunker(3)` | 8 | Không cắt giữa câu; Q5 có điểm cao nhất (0.681) vì chunk 3 câu rất ngắn khớp đúng câu "Điểm ĐATN…" | Không biết ranh giới Khoản: các điểm `a); b); c)` kết thúc bằng `;` bị coi là một câu → chunk dài bất thường (max 780); Q2 miss |

**Ablation của `SectionChunker` (cùng 5 câu hỏi) — vì sao chọn v2 với `max_chars=900, min_chars=250`:**

| Biến thể | Số chunk | Q1 | Q2 | Q3 | Q4 | Q5 | Tổng |
|---|---|---|---|---|---|---|---|
| v1: breadcrumb ghép vào nội dung, `min_chars=80` | 55 | 2 | 1 (top-3) | 1 (top-2) | 2 | 2 | 8 |
| v1: breadcrumb ghép vào nội dung, `min_chars=250` | 36 | 2 | 1 (top-2) | 1 (top-2) | 2 | 2 | 8 |
| v2: breadcrumb → metadata, `min_chars=80` | 55 | 2 | **0 (miss)** | 2 | 2 | 2 | 8 |
| **v2: breadcrumb → metadata, `min_chars=250` (chọn)** | 36 | 2 | 1 (top-2) | 2 | 2 | 2 | **9** |
| v2, `max_chars=500` | 43 | 2 | 1 (top-2) | 2 | 2 | 2 | 9 |

- **Breadcrumb trong nội dung làm loãng embedding (v1 → v2):** mọi chunk của Điều 16 đều bắt đầu bằng cùng tiền tố "Điều 16. Nghỉ học tạm thời và tự nguyện thôi học › …" nên chúng giống nhau hơn và chunk định nghĩa (Khoản 1) lấn át chunk chứa thời hạn (Khoản 2) ở Q3. Đưa breadcrumb sang `metadata["section"]` giữ được lợi ích trích dẫn mà không trả giá này.
- **Gộp Khoản ngắn (`min_chars=250`) cứu Q2:** không gộp, chunk "Khoản 3. Buộc thôi học" đứng riêng, ngắn, bị các chunk "tự nguyện thôi học" của Điều 16 đẩy khỏi top-3; gộp Khoản 2 + Khoản 3 của Điều 19 lại thì chunk có thêm ngữ cảnh "cảnh báo học tập" và lên top-2.
- `max_chars=500` cho kết quả bằng nhau (9) nhưng nhiều chunk hơn (43 so với 36) mà không thêm điểm nào; nhóm giữ 900 để một Khoản liệt kê điều kiện (ví dụ 4 điều kiện tốt nghiệp ở Điều 14 Khoản 3) luôn nằm trong **một** chunk khi agent đọc.

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> Về điểm truy xuất thuần, bốn chiến lược chỉ chênh nhau 1 điểm (8–9/10) vì corpus nhỏ và 3/5 câu hỏi có từ khoá rất đặc trưng; điểm thắng của hai chiến lược nhận biết Khoản (Section của Hùng, Recursive tinh chỉnh của Hưởng) đều đến từ câu khó Q2/Q3, nơi chunk phải chứa **đủ câu dẫn + toàn bộ Khoản**. Nhóm chọn **`SectionChunker` v2 làm chiến lược chính** vì: (1) điểm cao nhất và là chiến lược duy nhất đưa Q3 lên top-1 nhờ giữ nguyên khối Khoản 2 Điều 16; (2) ít chunk nhất (36 so với 45–49) và 100% chunk là một quy định trọn vẹn → agent đọc đúng một Khoản, không phải ghép mảnh; (3) `metadata["section"]` cho phép trả lời kèm "theo Điều 16 Khoản 2" — đúng cách người dùng quy chế cần. Bài học chung của nhóm: với văn bản quy chế, **ranh giới ngữ nghĩa (Khoản) quan trọng hơn kích thước chunk**, và ngữ cảnh phụ (breadcrumb) nên đi vào metadata chứ không nhồi vào nội dung được embed.

**Failure case còn tồn tại ở mọi chiến lược (đưa vào mục 3.5 / phân tích lỗi):** Q2 "Khi nào sinh viên bị buộc thôi học?" — top-1 của cả 4 chiến lược đều rơi vào Điều 16 (Khoản 4 "khi chế độ nghỉ tạm thời có hiệu lực…" với Section/Recursive, Khoản 5 "Tự nguyện thôi học" với Sentence) chứ không phải Khoản 3 Điều 19 "Buộc thôi học", vì embedding đa ngữ MiniLM bám vào cụm "thôi học"/"nghỉ học" chứ không phân biệt "buộc" với "tự nguyện". Đây là giới hạn của embedder chứ không phải chunking; hướng cải thiện là lọc `metadata_filter={"category": "academic-warning"}` hoặc thêm trường `keywords` khi ingest.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

> **Đúng 5 câu hỏi**, đa dạng, có thể kiểm chứng; **ít nhất 1 câu** cần lọc metadata mới trả lời tốt. Đây là bộ câu hỏi chung cho mọi thành viên chạy.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong học kỳ hè? | **Tối đa 8 TC** trong học kỳ hè (học kỳ chính: tối đa 24 TC, tối thiểu 12 TC nếu không bị cảnh báo học tập). | `dieu-10-dang-ky-hoc-tap` — Điều 10, Khoản 2, điểm a |
| 2 | Khi nào sinh viên bị buộc thôi học? | Khi thuộc một trong hai trường hợp: (a) bị **cảnh báo học tập mức 3 lần thứ hai liên tiếp**; (b) học chậm tiến độ quá thời gian cho phép hoặc không còn đủ khả năng tốt nghiệp trong thời gian cho phép (khoản 3 Điều 3). | `dieu-19-20-canh-bao-hoc-tap-va-buoc-thoi-hoc` — Điều 19, Khoản 3 |
| 3 | Nghỉ học tạm thời vì lý do cá nhân thì được nghỉ tối đa bao lâu? | Với lý do khác ốm đau/thai sản/tai nạn/lực lượng vũ trang: sinh viên phải đã học ít nhất một học kỳ; **nghỉ tối đa 04 học kỳ chính**, thời gian này tính vào thời gian học chậm tiến độ; nghỉ quá 04 học kỳ chính sẽ bị xét buộc thôi học. | `dieu-16-nghi-hoc-tam-thoi-va-thoi-hoc` — Điều 16, Khoản 2, điểm d |
| 4 | Điều kiện để được xét công nhận tốt nghiệp là gì? | Đủ 4 điều kiện: (a) hoàn thành đầy đủ học phần của CTĐT trong thời gian quy định, kể cả GDTC và GDQP-AN; (b) **đạt chuẩn ngoại ngữ đầu ra**; (c) điểm trung bình tích lũy toàn khóa **≥ 2,0**; (d) không bị truy cứu trách nhiệm hình sự / không đang bị kỷ luật đình chỉ học tập. | `dieu-14-15-dang-ky-tot-nghiep-va-hang-tot-nghiep` — Điều 14, Khoản 3 |
| 5 | Điểm ĐATN được tính từ điểm quá trình và điểm cuối kỳ theo trọng số nào? **(chạy với `metadata_filter={"audience": "student"}`)** | Điểm ĐATN = **0,5 × điểm quá trình + 0,5 × điểm cuối kỳ**. | `dieu-13-dieu-kien-lam-do-an-tot-nghiep` (audience = student) — Điều 13, Khoản 2, điểm a. *Không lọc, top-1 là `dieu-13-cham-diem-do-an-tot-nghiep` (audience = faculty, hướng dẫn chấm cho hội đồng) — với Recursive cả top-3 đều là tài liệu faculty.* |

**Tính đa dạng của bộ câu hỏi:** Q1 hỏi một con số; Q2 hỏi điều kiện "khi nào" (nhiều trường hợp); Q3 hỏi thời hạn kèm điều kiện áp dụng; Q4 hỏi danh sách điều kiện; Q5 hỏi công thức và bắt buộc lọc `audience`. Năm câu rơi vào 5 tài liệu khác nhau (Điều 10, 19, 16, 14, 13). Mọi gold answer đều trích nguyên văn từ corpus, không suy đoán.

### Tổng hợp chất lượng truy xuất của nhóm

> Cách chấm (theo `docs/SCORING.md`): **2 điểm/câu** — top-3 chứa chunk liên quan + agent trả lời đúng (2), có liên quan nhưng thiếu/không ở top-1 (1), không có trong top-3 (0).

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> *Viết 2-3 câu:*

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
> *Liệt kê 2-3 ý:*

**Bài học rút ra khi so sánh trong nhóm:**
> *Viết 2-3 câu — cùng tài liệu nhưng chiến lược khác nhau dẫn tới khác biệt gì?*

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | / 10 |
| Thiết kế chiến lược (Strategy Design) | / 15 |
| Chất lượng truy xuất (Retrieval Quality) | / 10 |
| Thuyết trình (Demo) | / 5 |
| **Tổng phần nhóm** | **/ 40** |
