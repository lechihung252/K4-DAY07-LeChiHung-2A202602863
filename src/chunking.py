from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+|(?<=\.)\n", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        for start in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[start : start + self.max_sentences_per_chunk]
            chunks.append(" ".join(group).strip())
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators or remaining_separators[0] == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        sep = remaining_separators[0]
        rest_separators = remaining_separators[1:]

        parts = [p for p in current_text.split(sep) if p]
        pieces: list[str] = []
        for part in parts:
            if len(part) > self.chunk_size:
                pieces.extend(self._split(part, rest_separators))
            else:
                pieces.append(part)

        merged: list[str] = []
        current = ""
        for piece in pieces:
            candidate = current + sep + piece if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = piece
        if current:
            merged.append(current)
        return merged


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        result = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = (sum(len(c) for c in chunks) / count) if count else 0.0
            result[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks,
            }
        return result


class SectionChunker:
    """
    Chia nhỏ văn bản quy chế/sổ tay theo tiêu đề Markdown (Điều / Khoản).

    Lý do thiết kế: mỗi Khoản trong quy chế đào tạo là một đơn vị ý nghĩa
    trọn vẹn (một quy định, một điều kiện). Cắt theo ranh giới này giữ được
    ngữ cảnh pháp lý, và gắn "breadcrumb" (tên Điều › tên Khoản) vào đầu mỗi
    chunk để embedding biết chunk đang nói về điều gì ngay cả khi nội dung
    chỉ là danh sách a), b), c).

    Rules:
        - Bỏ YAML frontmatter (--- ... ---) nếu có.
        - Cắt tại mọi dòng tiêu đề "# ..." hoặc "## ...".
        - Khoản dài hơn max_chars được tách tiếp theo các điểm a), b), c)
          (mỗi mảnh giữ lại câu dẫn của Khoản).
        - Khoản ngắn hơn min_chars được gộp vào chunk liền trước.
        - chunk_with_sections() trả về (breadcrumb, nội dung) để đưa breadcrumb
          vào metadata["section"]; chunk() chỉ ghép breadcrumb vào nội dung khi
          include_breadcrumb=True (v1 — thử nghiệm cho thấy làm loãng embedding).
    """

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
    FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
    ITEM_RE = re.compile(r"\n(?=[a-zđ]\)\s)")

    def __init__(
        self, max_chars: int = 900, min_chars: int = 250, include_breadcrumb: bool = False
    ) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.include_breadcrumb = include_breadcrumb

    def chunk(self, text: str) -> list[str]:
        pairs = self.chunk_with_sections(text)
        if self.include_breadcrumb:
            return [f"{crumb}\n{body}".strip() if crumb else body for crumb, body in pairs]
        return [body for _, body in pairs]

    def chunk_with_sections(self, text: str) -> list[tuple[str, str]]:
        """Trả về [(breadcrumb "Điều X › Khoản Y", nội dung chunk)]."""
        if not text:
            return []
        text = self.FRONTMATTER_RE.sub("", text).strip()
        if not text:
            return []

        pieces: list[tuple[str, str]] = []
        for breadcrumb, body in self._split_sections(text):
            if not body:
                continue
            for piece in self._fit(body):
                pieces.append((breadcrumb, piece))

        merged: list[tuple[str, str]] = []
        for crumb, body in pieces:
            if merged and len(body) < self.min_chars:
                prev_crumb, prev_body = merged[-1]
                merged[-1] = (prev_crumb, prev_body + "\n" + body)
            else:
                merged.append((crumb, body))
        return merged

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """Trả về [(breadcrumb, body)] cho từng tiêu đề."""
        matches = list(self.HEADING_RE.finditer(text))
        if not matches:
            return [("", text.strip())]

        sections: list[tuple[str, str]] = []
        trail: dict[int, str] = {}
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        for i, m in enumerate(matches):
            level = len(m.group(1))
            title = m.group(2).strip()
            trail = {lvl: t for lvl, t in trail.items() if lvl < level}
            trail[level] = title
            breadcrumb = " › ".join(trail[lvl] for lvl in sorted(trail))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[m.end() : end].strip()
            sections.append((breadcrumb, body))
        return sections

    def _fit(self, body: str) -> list[str]:
        if len(body) <= self.max_chars:
            return [body]
        parts = [p.strip() for p in self.ITEM_RE.split(body) if p.strip()]
        if len(parts) <= 1:
            return RecursiveChunker(chunk_size=self.max_chars).chunk(body)

        pieces: list[str] = []
        lead = parts[0]  # câu dẫn ("... được quy định như sau:") giữ lại cho từng mảnh
        current = lead
        for part in parts[1:]:
            candidate = current + "\n" + part
            if len(candidate) <= self.max_chars:
                current = candidate
            else:
                pieces.append(current)
                current = lead + "\n" + part if len(lead) < self.max_chars // 2 else part
        pieces.append(current)
        return pieces
