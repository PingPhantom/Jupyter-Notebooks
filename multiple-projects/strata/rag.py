"""StrataRAG — คลังความรู้สำหรับฉีดเข้าพรอมป์ (chunking, embedding, hybrid search, บันทึก/โหลด)."""
from __future__ import annotations

import hashlib
import io
import json
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import DEFAULT_RAG_PATH

if TYPE_CHECKING:  # fastembed / pypdf ถูก import ตอนใช้จริง (ประหยัดเวลา import + ไม่บังคับติดตั้ง)
    from fastembed import TextEmbedding


def _safe_print(text: str) -> None:
    """แสดงข้อความแบบ thread-safe บน Colab / Jupyter ไม่ให้แครชเมื่อเรียกจาก Background Thread."""
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        from rich import print as rprint

        rprint(text)
    except Exception:
        clean = re.sub(r"\[.*?\]", "", text)
        print(clean)


class StrataRAG:
    """คลังความรู้ RAG: ตัด chunk ที่ขอบประโยค, embedding แบบ lazy, hybrid search (เวกเตอร์ + คีย์เวิร์ด), บันทึก/โหลดได้"""

    def __init__(self, embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.embedding_model = embedding_model
        self._embedder: TextEmbedding | None = None
        self.chunks: list[dict] = []
        self._parts: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    @property
    def embedder(self) -> TextEmbedding:
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    from fastembed import TextEmbedding

                    self._embedder = TextEmbedding(model_name=self.embedding_model)
        return self._embedder

    @property
    def embeddings(self) -> np.ndarray | None:
        if self._matrix is None and self._parts:
            self._matrix = np.concatenate(self._parts, axis=0) if len(self._parts) > 1 else self._parts[0]
        return self._matrix

    @embeddings.setter
    def embeddings(self, value: np.ndarray | None):
        self._parts = [value] if value is not None and len(value) else []
        self._matrix = self._parts[0] if self._parts else None

    @staticmethod
    def _sig(source: str, text: str) -> str:
        return hashlib.sha1(f"{source}|{text}".encode("utf-8", errors="ignore")).hexdigest()

    def clear(self):
        with self._lock:
            self.chunks.clear()
            self._parts.clear()
            self._matrix = None
            self._seen.clear()

    @staticmethod
    def split_chunks(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
        """แบ่งข้อความโดยตัดที่ขอบย่อหน้า/ประโยคก่อน ไม่ตัดกลางคำ และไม่ปล่อย chunk เล็กจิ๋ว"""
        text = re.sub(r"[ \t]+", " ", str(text).replace("\r\n", "\n")).strip()
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]
        units = [u.strip() for u in re.split(r"(?<=[.!?。！？:;])\s+|\n{2,}|\n", text) if u and u.strip()]
        chunks: list[str] = []
        buf = ""
        for u in units:
            while len(u) > chunk_size:  # ประโยคเดียวยาวเกิน chunk -> ตัดเป็นท่อน ๆ
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.append(u[:chunk_size].strip())
                rest = u[max(0, chunk_size - overlap):]
                sp = rest.find(" ") if overlap > 0 else -1
                u = (rest[sp + 1:] if sp != -1 else rest).strip()
            if not u:
                continue
            if not buf:
                buf = u
            elif len(buf) + 1 + len(u) <= chunk_size:
                buf = f"{buf} {u}"
            else:
                chunks.append(buf)
                tail = ""
                if overlap > 0:  # overlap ต้องเริ่มที่ขอบคำ ไม่ตัดกลางคำ
                    tail_raw = buf[-overlap:]
                    sp = tail_raw.find(" ")
                    tail = (tail_raw[sp + 1:] if sp != -1 else tail_raw).strip()
                buf = f"{tail} {u}".strip() if tail else u
        if buf.strip():
            chunks.append(buf.strip())
        return [c for c in chunks if c.strip()]

    def add_documents(self, docs: list[str | dict], chunk_size: int = 500, overlap: int = 80, max_chunks: int = 4000) -> int:
        with self._lock:
            new_c: list[dict] = []
            for idx, doc in enumerate(docs):
                txt = doc.get("text", "") if isinstance(doc, dict) else str(doc)
                src = (doc.get("source") if isinstance(doc, dict) else None) or f"doc_{len(self.chunks) + idx + 1}"
                for pc in self.split_chunks(txt, chunk_size=chunk_size, overlap=overlap):
                    sig = self._sig(src, pc)
                    if sig in self._seen:
                        continue
                    self._seen.add(sig)
                    new_c.append({"text": pc, "source": src})
            if not new_c:
                _safe_print("[yellow]ℹ️ ไม่มีเนื้อหาใหม่ (ซ้ำกับที่มีอยู่แล้ว)[/yellow]")
                return 0
            if len(self.chunks) + len(new_c) > max_chunks:
                raise RuntimeError(f"คลัง RAG จะเกิน {max_chunks} chunks — ล้างคลังหรือแบ่งไฟล์ก่อนนำเข้า")
            mat = np.asarray(list(self.embedder.embed([c["text"] for c in new_c])), dtype=np.float32)
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
            self.chunks.extend(new_c)
            self._parts.append(mat)
            self._matrix = None
            _safe_print(f"[bold green]📚 เพิ่มเข้า StrataRAG สำเร็จ:[/bold green] +{len(new_c)} chunks (รวม {len(self.chunks)})")
            return len(new_c)

    def add_pdf_bytes(self, data: bytes, filename: str = "document.pdf", chunk_size: int = 500, overlap: int = 80) -> int:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        docs, empty = [], 0
        for pno, pg in enumerate(reader.pages, 1):
            txt = (pg.extract_text() or "").strip()
            if txt:
                docs.append({"source": f"{filename} (หน้า {pno})", "text": txt})
            else:
                empty += 1
        if empty:
            _safe_print(f"[dim]   (ข้าม {empty} หน้าที่ไม่มีข้อความ — อาจเป็นหน้าสแกนรูป)[/dim]")
        return self.add_documents(docs, chunk_size=chunk_size, overlap=overlap)

    def add_files(self, *file_paths: str | Path, chunk_size: int = 500, overlap: int = 80, max_file_mb: float = 25.0) -> int:
        txt_docs, added = [], 0
        for fp in file_paths:
            p = Path(fp)
            if not p.is_file():
                continue
            if p.stat().st_size > max_file_mb * 1024**2:
                _safe_print(f"[yellow]⚠️ ข้าม {p.name}: ใหญ่เกิน {max_file_mb:.0f} MB[/yellow]")
                continue
            if p.suffix.lower() == ".pdf":
                added += self.add_pdf_bytes(p.read_bytes(), filename=p.name, chunk_size=chunk_size, overlap=overlap)
            else:
                txt_docs.append({"source": p.name, "text": p.read_text(encoding="utf-8", errors="ignore")})
        if txt_docs:
            added += self.add_documents(txt_docs, chunk_size=chunk_size, overlap=overlap)
        return added

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05, hybrid_weight: float = 0.25) -> list[dict]:
        """ค้นหาแบบ Hybrid Search: ผสานเวกเตอร์ Cosine Similarity + คีย์เวิร์ด Lexical Overlap เพื่อความแม่นยำสูงสุด"""
        emb = self.embeddings
        q = str(query).strip()
        if emb is None or not self.chunks or not q:
            return []
        qv = np.asarray(next(iter(self.embedder.embed([q]))), dtype=np.float32)
        qv /= np.linalg.norm(qv) + 1e-10
        vec_sims = (emb @ qv).flatten()

        # คำนวณ Lexical Score เพื่อเสริมคำศัพท์เทคนิค / โค้ด / ชื่อเฉพาะ
        q_tokens = [w.lower() for w in re.findall(r"[\w]+", q) if len(w) >= 2]
        q_lower = q.lower()
        n_chunks = len(self.chunks)
        lex_scores = np.zeros(n_chunks, dtype=np.float32)

        if q_tokens:
            for idx, c in enumerate(self.chunks):
                txt_low = c["text"].lower()
                matches = sum(1 for tok in q_tokens if tok in txt_low)
                ratio = matches / len(q_tokens)
                phrase_bonus = 0.35 if (len(q_tokens) >= 2 and q_lower in txt_low) else 0.0
                lex_scores[idx] = min(1.0, ratio + phrase_bonus)

        # ผสานน้ำหนัก Hybrid: Vector (75%) + Lexical (25%)
        combined = (1.0 - hybrid_weight) * vec_sims + hybrid_weight * lex_scores
        order = np.argsort(-combined)[: max(1, int(top_k))]
        hits = []
        for idx in order:
            score = float(combined[int(idx)])
            if score < min_score:
                continue
            hits.append({
                "rank": len(hits) + 1,
                "score": round(score, 4),
                "vec_score": round(float(vec_sims[int(idx)]), 4),
                "lex_score": round(float(lex_scores[int(idx)]), 4),
                "source": self.chunks[int(idx)]["source"],
                "text": self.chunks[int(idx)]["text"],
            })
        return hits

    def build_augmented_prompt(self, question: str, top_k: int = 3, min_score: float = 0.05) -> tuple[str, list[dict]]:
        hits = self.search(question, top_k=top_k, min_score=min_score)
        if not hits:
            return question, []
        ctx = "\n\n".join(f"[{h['source']} | score={h['score']}]\n{h['text']}" for h in hits)
        return f"ใช้ข้อมูลอ้างอิงต่อไปนี้ตอบคำถามและระบุแหล่งที่มา ([source]):\n\n--- ข้อมูลอ้างอิง ---\n{ctx}\n-------------------\n\nคำถาม: {question}", hits

    def save(self, path: str | Path = DEFAULT_RAG_PATH) -> str:
        with self._lock:
            emb = self.embeddings
            if emb is None or not self.chunks:
                raise RuntimeError("คลังว่างเปล่า ไม่มีอะไรให้บันทึก")
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                p,
                embeddings=emb.astype(np.float32),
                chunks=np.asarray(json.dumps(self.chunks, ensure_ascii=False)),
                model=np.asarray(self.embedding_model),
            )
            _safe_print(f"[bold green]💾 บันทึกคลัง RAG แล้ว:[/bold green] {p} ({len(self.chunks)} chunks)")
            return str(p)

    def load(self, path: str | Path = DEFAULT_RAG_PATH) -> int:
        with self._lock:
            p = Path(path)
            if not p.is_file():
                raise FileNotFoundError(f"ไม่พบไฟล์คลัง: {p}")
            try:
                with np.load(p, allow_pickle=False) as z:
                    emb = np.asarray(z["embeddings"], dtype=np.float32)
                    chunks = json.loads(str(z["chunks"].item() if hasattr(z["chunks"], "item") else z["chunks"]))
            except Exception as e:
                raise RuntimeError(f"อ่านไฟล์คลังไม่สำเร็จ ({p}): {e}") from e
            if not isinstance(chunks, list) or emb.ndim != 2 or emb.shape[0] != len(chunks):
                raise RuntimeError(f"ไฟล์คลังเสียหาย: embeddings {getattr(emb, 'shape', '?')} vs chunks {len(chunks) if isinstance(chunks, list) else '?'}")
            self.chunks = chunks
            self.embeddings = (emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)).astype(np.float32)
            self._seen = {self._sig(c.get("source", ""), c.get("text", "")) for c in chunks}
            _safe_print(f"[bold green]📂 โหลดคลัง RAG แล้ว:[/bold green] {len(chunks)} chunks จาก {p}")
            return len(chunks)


# คลังกลางที่ UI ใช้ (สร้าง lazy เพื่อไม่ให้ import แล้วโหลด embedding model ทันที)
GLOBAL_RAG = StrataRAG()
