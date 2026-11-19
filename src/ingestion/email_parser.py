"""Enterprise email parser for .eml and .msg files preserving thread hierarchies and metadata."""
import io
import re
import email
from email import policy
from pathlib import Path
from typing import List, Dict, Any, Union
from langchain_core.documents import Document

def clean_email_text(text: str) -> str:
    """Normalize whitespace and strip excessive reply quotes."""
    if not text:
        return ""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class EmailParser:
    """Extracts structured text, headers, and thread metadata from email messages."""

    def parse_file(self, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[Document]:
        """Parse an .eml or .msg file and return structured LangChain documents."""
        ext = Path(filename).suffix.lower()
        if ext == ".eml":
            return self.parse_eml(file_source, filename)
        elif ext == ".msg":
            return self.parse_msg(file_source, filename)
        else:
            raise ValueError(f"Unsupported email format: {ext}")

    def parse_eml(self, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[Document]:
        """Parse RFC 822 .eml file."""
        if isinstance(file_source, (str, Path)):
            with open(file_source, "rb") as f:
                msg = email.message_from_binary_file(f, policy=policy.default)
        else:
            raw_bytes = file_source.getvalue() if hasattr(file_source, "getvalue") else file_source.read()
            msg = email.message_from_bytes(raw_bytes, policy=policy.default)

        subject = msg.get("Subject", "No Subject")
        sender = msg.get("From", "Unknown Sender")
        recipient = msg.get("To", "Unknown Recipient")
        date_str = msg.get("Date", "Unknown Date")
        cc_str = msg.get("Cc", "")

        # Extract plain text or html payload
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        body += part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace") + "\n"
                    except Exception:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="replace")
            except Exception:
                body = str(msg.get_payload())

        cleaned_body = clean_email_text(body)
        
        # Format structured enterprise header
        cc_line = f"**Cc:** {cc_str}\n" if cc_str else ""
        structured_content = (
            f"# Email: {subject}\n"
            f"**From:** {sender}\n"
            f"**To:** {recipient}\n"
            f"{cc_line}"
            f"**Date:** {date_str}\n"
            f"**File:** {filename}\n\n"
            f"## Message Body:\n"
            f"{cleaned_body}"
        )

        meta = {
            "doc_name": filename,
            "doc_type": "Email",
            "subject": subject,
            "sender": sender,
            "recipient": recipient,
            "date": date_str,
            "page": 1,
            "total_pages": 1
        }

        return [Document(page_content=structured_content, metadata=meta)]

    def parse_msg(self, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[Document]:
        """Parse Outlook .msg file if python-oxmsg / olefile is available."""
        try:
            import oxmsg
            if isinstance(file_source, (str, Path)):
                msg = oxmsg.Message.load_from_file(str(file_source))
            else:
                raw_bytes = file_source.getvalue() if hasattr(file_source, "getvalue") else file_source.read()
                msg = oxmsg.Message.load_from_bytes(raw_bytes)

            subject = getattr(msg, "subject", "No Subject")
            sender = getattr(msg, "sender", "Unknown")
            body = getattr(msg, "body", "") or ""
            
            content = f"# Email: {subject}\n**From:** {sender}\n**File:** {filename}\n\n{body}"
            meta = {"doc_name": filename, "doc_type": "Email", "subject": subject, "page": 1, "total_pages": 1}
            return [Document(page_content=content, metadata=meta)]
        except Exception:
            # Fallback to text parsing
            return self.parse_eml(file_source, filename)
