"""Multi-modal document dispatcher for enterprise documents (PDF, PPTX, EML, MSG, DOCX)."""
import io
import logging
from typing import List, Union
from pathlib import Path
from langchain_core.documents import Document

from src.ingestion.docling_parser import DoclingParser
from src.ingestion.email_parser import EmailParser

logger = logging.getLogger(__name__)

class EnterpriseDocumentParser:
    """Unified ingestion pipeline supporting Docling, Email Parsing, and Naive baselines."""

    def __init__(self, parser_engine: str = "docling", enable_ocr: bool = True):
        self.parser_engine = parser_engine
        self.enable_ocr = enable_ocr
        self.docling_parser = DoclingParser(enable_ocr=enable_ocr)
        self.email_parser = EmailParser()

    def parse_file(
        self,
        file_source: Union[str, Path, io.BytesIO],
        filename: str,
        force_naive: bool = False
    ) -> List[Document]:
        """Parse any enterprise document format."""
        ext = Path(filename).suffix.lower()

        # Handle emails
        if ext in [".eml", ".msg"]:
            return self.email_parser.parse_file(file_source, filename)

        # Handle Naive baseline for comparison
        if force_naive or self.parser_engine == "naive_pymupdf":
            return self.docling_parser._fallback_parse(Path(file_source) if isinstance(file_source, (str, Path)) else file_source, filename, ext)

        # Handle through Docling
        return self.docling_parser.parse_file(file_source, filename)
