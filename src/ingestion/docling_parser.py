"""IBM Docling ingestion engine for parsing messy enterprise PDFs, tables, slides, and scans."""
import io
import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DoclingParser:
    """Document parser powered by IBM Docling for deep layout, table, and multi-modal extraction."""

    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions

                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = self.enable_ocr
                pipeline_options.do_table_structure = True

                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
                logger.info("Initialized IBM Docling DocumentConverter with table structure and OCR.")
            except Exception as e:
                logger.warning(f"Failed to initialize full Docling converter: {e}. Using lightweight fallback.")
                self._converter = False
        return self._converter

    def parse_file(self, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[Document]:
        """Convert a document into structured markdown documents using IBM Docling."""
        ext = Path(filename).suffix.lower()
        converter = self._get_converter()

        # If file_source is BytesIO, write to temporary file because Docling accepts filepaths
        temp_path = None
        if isinstance(file_source, (str, Path)):
            input_path = Path(file_source)
        else:
            import tempfile
            file_bytes = file_source.getvalue() if hasattr(file_source, "getvalue") else file_source.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
                tf.write(file_bytes)
                temp_path = Path(tf.name)
            input_path = temp_path

        docs: List[Document] = []
        try:
            if converter:
                try:
                    conv_res = converter.convert(input_path)
                    docling_doc = conv_res.document
                    
                    # 1. Export whole document as markdown preserving tables and hierarchy
                    full_md = docling_doc.export_to_markdown()
                    
                    # 2. Check if pages are available for page-level attribution
                    num_pages = len(getattr(docling_doc, "pages", []))
                    if num_pages > 1:
                        # Extract per-page markdown if available, else split by page anchors
                        sections = full_md.split("\n<!-- page_break -->\n") if "<!-- page_break -->" in full_md else [full_md]
                        for idx, sec in enumerate(sections, start=1):
                            if sec.strip():
                                docs.append(Document(
                                    page_content=sec.strip(),
                                    metadata={
                                        "doc_name": filename,
                                        "doc_type": "PDF (Docling Structured)" if ext == ".pdf" else "Document (Docling)",
                                        "parser": "IBM_Docling",
                                        "page": idx,
                                        "total_pages": len(sections),
                                        "tables_detected": full_md.count("|---")
                                    }
                                ))
                    else:
                        docs.append(Document(
                            page_content=full_md.strip(),
                            metadata={
                                "doc_name": filename,
                                "doc_type": "PDF (Docling Structured)" if ext == ".pdf" else "Document (Docling)",
                                "parser": "IBM_Docling",
                                "page": 1,
                                "total_pages": 1,
                                "tables_detected": full_md.count("|---")
                            }
                        ))
                    logger.info(f"Docling parsed '{filename}' into {len(docs)} structured page units.")
                    return docs
                except Exception as e:
                    logger.warning(f"Docling conversion exception for {filename}: {e}. Falling back to PyMuPDF.")

            # Fallback to PyMuPDF / python-pptx if Docling encountered an issue or is disabled
            docs = self._fallback_parse(input_path, filename, ext)
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

        return docs

    def _fallback_parse(self, input_path: Path, filename: str, ext: str) -> List[Document]:
        """Fallback extraction using PyMuPDF or text reader."""
        docs = []
        if ext == ".pdf":
            import pymupdf
            doc = pymupdf.open(str(input_path))
            for idx, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "doc_name": filename,
                            "doc_type": "PDF (Naive)",
                            "parser": "PyMuPDF_Naive",
                            "page": idx + 1,
                            "total_pages": len(doc)
                        }
                    ))
        elif ext in [".pptx", ".ppt"]:
            from pptx import Presentation
            prs = Presentation(str(input_path))
            for idx, slide in enumerate(prs.slides):
                slide_texts = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        slide_texts.append(shape.text_frame.text.strip())
                content = "\n".join([t for t in slide_texts if t])
                if content:
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "doc_name": filename,
                            "doc_type": "Presentation",
                            "parser": "python-pptx",
                            "page": idx + 1,
                            "total_pages": len(prs.slides)
                        }
                    ))
        elif ext in [".txt", ".md"]:
            with open(input_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={"doc_name": filename, "doc_type": "Text", "page": 1, "total_pages": 1}
                ))
        return docs
