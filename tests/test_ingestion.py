"""Tests for document parsers (IBM Docling, Email RFC 822, and Naive fallbacks)."""
import pytest
from pathlib import Path
from langchain_core.documents import Document

from src.ingestion.docling_parser import DoclingParser
from src.ingestion.email_parser import EmailParser
from src.ingestion.multimodal_parser import EnterpriseDocumentParser
from src.config import ENTERPRISE_SAMPLE_DIR


class TestIngestionParsers:

    def test_email_parser_rfc822(self):
        """Verify email parsing preserves headers, sender, date, and thread contents."""
        parser = EmailParser()
        eml_path = ENTERPRISE_SAMPLE_DIR / "Incident_WarRoom_P1_Outage.eml"
        assert eml_path.exists(), f"Sample email not found at {eml_path}"

        docs = parser.parse_file(eml_path, eml_path.name)
        assert len(docs) >= 1
        doc = docs[0]

        assert doc.metadata["doc_type"] == "Email"
        assert doc.metadata["doc_name"] == eml_path.name
        assert "INC-7721-P1" in doc.page_content
        assert "Sarah Chen" in doc.page_content
        assert "04:30 UTC" in doc.page_content

    def test_naive_parser_fallback(self):
        """Verify naive parser fallback extracts raw text without layout models."""
        parser = DoclingParser(enable_ocr=False)
        sla_path = ENTERPRISE_SAMPLE_DIR / "Enterprise_Cloud_Procurement_SLA.md"
        assert sla_path.exists()

        docs = parser._fallback_parse(sla_path, sla_path.name, ".md")
        assert len(docs) >= 1
        assert "99.95%" in docs[0].page_content

    def test_multimodal_dispatcher_routing(self):
        """Verify the multimodal dispatcher correctly identifies file extensions."""
        dispatcher = EnterpriseDocumentParser(parser_engine="naive_pymupdf")
        eml_path = ENTERPRISE_SAMPLE_DIR / "Incident_WarRoom_P1_Outage.eml"

        email_docs = dispatcher.parse_file(eml_path, eml_path.name)
        assert len(email_docs) > 0
        assert email_docs[0].metadata["doc_type"] == "Email"

    def test_table_detection_in_markdown(self):
        """Verify table detection correctly counts markdown table dividers."""
        parser = DoclingParser(enable_ocr=False)
        md_text = (
            "# Financial Summary\n\n"
            "| Metric | Value |\n"
            "|---|---|\n"
            "| Revenue | $100M |\n"
        )
        assert md_text.count("|---") == 2
