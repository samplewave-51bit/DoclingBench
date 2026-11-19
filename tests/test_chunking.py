"""Tests for chunking strategies (Docling Structural, Recursive, Semantic, Hierarchical)."""
import pytest
from langchain_core.documents import Document
from src.chunking.chunking_engine import MultiStrategyChunker


class TestChunkingEngine:

    @pytest.fixture
    def chunker(self):
        return MultiStrategyChunker()

    @pytest.fixture
    def document_with_table(self):
        return [
            Document(
                page_content=(
                    "# Executive Quarterly Report\n\n"
                    "This section discusses segment performance.\n\n"
                    "| Segment | Revenue | Margin |\n"
                    "| :--- | :--- | :--- |\n"
                    "| Cloud | $1840M | 28.4% |\n"
                    "| SaaS | $920M | 34.1% |\n\n"
                    "Conclusion and strategic recommendations follow."
                ),
                metadata={"doc_name": "Report.pdf", "page": 1}
            )
        ]

    def test_docling_structural_preserves_tables(self, chunker, document_with_table):
        """Docling structural chunking must keep table structures intact."""
        chunks = chunker.chunk_documents(document_with_table, strategy="docling_structural")
        assert len(chunks) >= 1

        table_chunks = [c for c in chunks if c.metadata.get("is_table", False)]
        assert len(table_chunks) >= 1
        # Table must contain both rows intact
        assert "Cloud" in table_chunks[0].page_content
        assert "SaaS" in table_chunks[0].page_content
        assert table_chunks[0].metadata["chunking_strategy"] == "docling_structural"

    def test_recursive_character_chunking(self, chunker, document_with_table):
        """Recursive chunking slices text into bounded windows."""
        chunks = chunker.chunk_documents(document_with_table, strategy="recursive_character")
        assert len(chunks) >= 1
        for c in chunks:
            assert c.metadata["chunking_strategy"] == "recursive_character"
            assert "chunk_id" in c.metadata

    def test_semantic_boundary_chunking(self, chunker, document_with_table):
        """Semantic boundary chunking splits on sentence/paragraph transitions."""
        chunks = chunker.chunk_documents(document_with_table, strategy="semantic_boundary")
        assert len(chunks) >= 1
        for c in chunks:
            assert c.metadata["chunking_strategy"] == "semantic_boundary"

    def test_hierarchical_parent_child_linking(self, chunker, document_with_table):
        """Hierarchical chunking creates leaf chunks referencing parent chunks."""
        chunks = chunker.chunk_documents(document_with_table, strategy="hierarchical_parent")
        assert len(chunks) >= 1
        has_parent_link = any("parent_id" in c.metadata or "parent_content" in c.metadata for c in chunks)
        assert has_parent_link
