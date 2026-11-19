"""Multi-strategy chunking engine for empirical RAG comparison."""
import re
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class MultiStrategyChunker:
    """Implements and benchmarks multiple chunking strategies for unstructured enterprise data."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_documents(self, documents: List[Document], strategy: str = "docling_structural") -> List[Document]:
        """Dispatch to the requested chunking strategy."""
        if strategy == "docling_structural":
            return self._chunk_docling_structural(documents)
        elif strategy == "recursive_character":
            return self._chunk_recursive_character(documents)
        elif strategy == "semantic_boundary":
            return self._chunk_semantic_boundary(documents)
        elif strategy == "hierarchical_parent":
            return self._chunk_hierarchical_parent(documents)
        else:
            logger.warning(f"Unknown strategy '{strategy}'. Defaulting to docling_structural.")
            return self._chunk_docling_structural(documents)

    def _chunk_recursive_character(self, documents: List[Document]) -> List[Document]:
        """Baseline naive character chunking (often slices tables and paragraphs)."""
        chunked = []
        counter = 0
        for doc in documents:
            splits = self.recursive_splitter.split_text(doc.page_content)
            for s_idx, split_text in enumerate(splits):
                counter += 1
                meta = doc.metadata.copy()
                meta["chunk_id"] = f"{meta.get('doc_name', 'doc')}_rec_{counter}"
                meta["chunking_strategy"] = "recursive_character"
                meta["chunk_index"] = counter
                chunked.append(Document(page_content=split_text, metadata=meta))
        return chunked

    def _chunk_docling_structural(self, documents: List[Document]) -> List[Document]:
        """Layout & table-preserving chunking. Keeps Markdown tables and headers intact."""
        chunked = []
        counter = 0

        for doc in documents:
            text = doc.page_content
            doc_name = doc.metadata.get("doc_name", "Document")
            page_num = doc.metadata.get("page", 1)

            # Detect and protect markdown tables
            table_pattern = re.compile(r'(\|.+?\|\n\|[-:| ]+\|\n(?:\|.+?\|\n?)+)', re.MULTILINE)
            tables = table_pattern.findall(text)
            
            # Split sections by markdown headers (##, ###) or double newlines
            sections = re.split(r'\n(?=#{1,4}\s)', text)
            
            for sec in sections:
                sec_text = sec.strip()
                if not sec_text:
                    continue

                # Extract section title if exists
                first_line = sec_text.split('\n')[0]
                section_title = first_line.replace('#', '').strip() if first_line.startswith('#') else f"Page {page_num}"

                header_prefix = f"[{doc_name} | {section_title}]\n"

                # If section fits within chunk limit or is a table, keep whole
                if len(sec_text) <= self.chunk_size * 1.5 or "|---" in sec_text:
                    counter += 1
                    meta = doc.metadata.copy()
                    meta["chunk_id"] = f"{doc_name}_struct_{counter}"
                    meta["chunking_strategy"] = "docling_structural"
                    meta["section_title"] = section_title
                    meta["is_table"] = "|---" in sec_text
                    chunked.append(Document(page_content=f"{header_prefix}{sec_text}", metadata=meta))
                else:
                    # Split while trying to avoid breaking tables
                    sub_splits = self.recursive_splitter.split_text(sec_text)
                    for sub_t in sub_splits:
                        counter += 1
                        meta = doc.metadata.copy()
                        meta["chunk_id"] = f"{doc_name}_struct_{counter}"
                        meta["chunking_strategy"] = "docling_structural"
                        meta["section_title"] = section_title
                        meta["is_table"] = "|---" in sub_t
                        chunked.append(Document(page_content=f"{header_prefix}{sub_t}", metadata=meta))

        return chunked

    def _chunk_semantic_boundary(self, documents: List[Document]) -> List[Document]:
        """Sentence boundary chunking that preserves logical proposition units."""
        chunked = []
        counter = 0

        for doc in documents:
            text = doc.page_content
            # Split sentences using punctuation boundaries
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = []
            current_len = 0

            for sent in sentences:
                sent_len = len(sent)
                if current_len + sent_len > self.chunk_size and current_chunk:
                    counter += 1
                    chunk_text = " ".join(current_chunk)
                    meta = doc.metadata.copy()
                    meta["chunk_id"] = f"{meta.get('doc_name', 'doc')}_sem_{counter}"
                    meta["chunking_strategy"] = "semantic_boundary"
                    chunked.append(Document(page_content=chunk_text, metadata=meta))
                    
                    # Small sentence-level overlap
                    current_chunk = [current_chunk[-1], sent] if len(current_chunk) > 1 else [sent]
                    current_len = sum(len(s) for s in current_chunk)
                else:
                    current_chunk.append(sent)
                    current_len += sent_len

            if current_chunk:
                counter += 1
                meta = doc.metadata.copy()
                meta["chunk_id"] = f"{meta.get('doc_name', 'doc')}_sem_{counter}"
                meta["chunking_strategy"] = "semantic_boundary"
                chunked.append(Document(page_content=" ".join(current_chunk), metadata=meta))

        return chunked

    def _chunk_hierarchical_parent(self, documents: List[Document]) -> List[Document]:
        """Hierarchical parent-child chunking: child chunks for search, parent context attached."""
        chunked = []
        counter = 0

        for doc in documents:
            parent_text = doc.page_content
            doc_name = doc.metadata.get("doc_name", "Document")
            
            # Child splitter is smaller (250 chars)
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
            children = child_splitter.split_text(parent_text)

            for c_idx, child in enumerate(children):
                counter += 1
                meta = doc.metadata.copy()
                meta["chunk_id"] = f"{doc_name}_child_{counter}"
                meta["chunking_strategy"] = "hierarchical_parent"
                meta["child_index"] = c_idx
                meta["parent_id"] = f"{doc_name}_parent_{doc.metadata.get('page', 1)}"
                meta["parent_content"] = parent_text[:1200]  # Store parent reference
                chunked.append(Document(page_content=child, metadata=meta))

        return chunked
