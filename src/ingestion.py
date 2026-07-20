"""
Ingestion layer.

Responsible for:
1. Loading raw SDS PDFs.
2. Enriching each page with useful metadata (chemical name, CAS number,
   section headers) extracted via regex, so we can later do metadata
   filtering at retrieval time.
3. Splitting into overlapping chunks with a RecursiveCharacterTextSplitter.
"""
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

SECTION_RE = re.compile(r"SECTION\s+(\d+):\s*([A-Z ,()/&-]+)")
CAS_RE = re.compile(r"CAS Number\s*[:\-]?\s*([0-9\-]+)")
CHEMICAL_NAME_RE = re.compile(r"Chemical Name\s*[:\-]?\s*(.+)")


def _extract_chemical_name(text: str, fallback: str) -> str:
    m = CHEMICAL_NAME_RE.search(text)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    return fallback


def _extract_cas(text: str) -> str:
    m = CAS_RE.search(text)
    return m.group(1).strip() if m else "unknown"


def _dominant_section(text: str) -> str:
    """Return the last SECTION header seen before/within this chunk of text."""
    matches = SECTION_RE.findall(text)
    if not matches:
        return "unspecified"
    num, name = matches[-1]
    return f"SECTION {num}: {name.strip()}"


def load_raw_documents(data_dir: Path = config.DATA_DIR) -> List[Document]:
    """Load every PDF in data_dir into LangChain Documents (one per page)."""
    docs: List[Document] = []
    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        full_text = "\n".join(p.page_content for p in pages)
        chemical_name = _extract_chemical_name(full_text, fallback=pdf_path.stem)
        cas_number = _extract_cas(full_text)

        for page in pages:
            page.metadata.update(
                {
                    "source_file": pdf_path.name,
                    "chemical_name": chemical_name,
                    "cas_number": cas_number,
                }
            )
            docs.append(page)
    return docs


def chunk_documents(
    docs: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """Recursive Character Text Splitter with overlap, plus section tagging."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["section"] = _dominant_section(chunk.page_content)
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source_file', 'doc')}_{i}"
    return chunks


def build_corpus(data_dir: Path = config.DATA_DIR) -> List[Document]:
    raw_docs = load_raw_documents(data_dir)
    return chunk_documents(raw_docs)


if __name__ == "__main__":
    corpus = build_corpus()
    print(f"Loaded {len(corpus)} chunks from {config.DATA_DIR}")
    for c in corpus[:3]:
        print("-" * 60)
        print(c.metadata)
        print(c.page_content[:200])
