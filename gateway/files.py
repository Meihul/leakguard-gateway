"""File uploads for /chat: in-memory multipart parsing, validation and text
extraction. Nothing here ever writes to disk — the request body is read
into memory with a hard size cap and parsed with python-multipart's
low-level parser (bypassing Starlette's form parser, which spools larger
uploads to a temp file), and pypdf / python-docx read from BytesIO.

Extracted text is returned as segments, each tagged with its page (PDF) or
line number (txt/csv lines, docx paragraphs and table rows), so findings
can point at where in the file they came from.
"""
import io
import zipfile
from pathlib import PurePath

from python_multipart.multipart import MultipartParser, parse_options_header

ALLOWED_EXTENSIONS = (".txt", ".pdf", ".docx", ".csv")
MAX_FILE_BYTES = 5 * 1024 * 1024
# Room for the prompt and multipart framing on top of the file itself.
MAX_BODY_BYTES = MAX_FILE_BYTES + 256 * 1024
# A .docx is a zip; refuse ones that would expand to something huge.
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
# Average non-whitespace characters per page below which a PDF is treated
# as scanned images with no real text layer.
MIN_PDF_CHARS_PER_PAGE = 20
MAX_PARTS = 10


class UploadError(Exception):
    """Invalid upload; the message is safe to show the user (HTTP 400)."""


def too_large_message():
    return f"File is too large. The maximum size is {MAX_FILE_BYTES // (1024 * 1024)} MB."


def parse_multipart(body: bytes, content_type: str):
    """Returns ({field: str}, file_or_None) where file is
    {"filename", "data"}. Entirely in memory."""
    ctype, params = parse_options_header(content_type)
    boundary = params.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise UploadError("Expected a multipart/form-data request.")

    parts = []
    state = {"field": b"", "value": b"", "headers": {}, "data": None}

    def on_part_begin():
        state["headers"] = {}
        state["data"] = bytearray()

    def on_header_field(data, start, end):
        state["field"] += data[start:end]

    def on_header_value(data, start, end):
        state["value"] += data[start:end]

    def on_header_end():
        state["headers"][state["field"].lower()] = state["value"]
        state["field"], state["value"] = b"", b""

    def on_part_data(data, start, end):
        state["data"] += data[start:end]

    def on_part_end():
        if len(parts) >= MAX_PARTS:
            raise UploadError("Too many form fields.")
        parts.append((state["headers"], bytes(state["data"])))

    parser = MultipartParser(boundary, callbacks={
        "on_part_begin": on_part_begin, "on_header_field": on_header_field,
        "on_header_value": on_header_value, "on_header_end": on_header_end,
        "on_part_data": on_part_data, "on_part_end": on_part_end,
    })
    try:
        parser.write(body)
        parser.finalize()
    except UploadError:
        raise
    except Exception:
        raise UploadError("Malformed multipart request.")

    fields, upload = {}, None
    for headers, data in parts:
        _, disp = parse_options_header(headers.get(b"content-disposition", b""))
        name = disp.get(b"name", b"").decode("utf-8", "replace")
        if b"filename" in disp:
            if name != "file":
                continue
            if upload is not None:
                raise UploadError("Only one file can be uploaded per message.")
            upload = {"filename": disp[b"filename"].decode("utf-8", "replace"), "data": data}
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, upload


def file_extension(filename):
    return PurePath(filename or "").suffix.lower()


def validate(filename, data):
    ext = file_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        shown = ext or "(no extension)"
        raise UploadError(f"Unsupported file type {shown}. "
                          f"Allowed types: {', '.join(ALLOWED_EXTENSIONS)}.")
    if len(data) > MAX_FILE_BYTES:
        raise UploadError(too_large_message())
    return ext


def _decode_text(data):
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _lines(text):
    return [("line", i, line) for i, line in enumerate(text.splitlines(), start=1)]


def _extract_pdf(data):
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise UploadError("This PDF is password-protected and could not be checked.")
        pages = [(page.extract_text() or "") for page in reader.pages]
    except UploadError:
        raise
    except Exception:
        raise UploadError("Could not read this .pdf file; it may be corrupted.")
    real_chars = sum(len("".join(p.split())) for p in pages)
    scanned = real_chars < MIN_PDF_CHARS_PER_PAGE * max(len(pages), 1)
    return [("page", i, text) for i, text in enumerate(pages, start=1)], scanned


def _extract_docx(data):
    from docx import Document
    from docx.table import Table
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if sum(info.file_size for info in zf.infolist()) > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise UploadError("This .docx file is too large once unpacked.")
        doc = Document(io.BytesIO(data))
        lines = []
        for block in doc.iter_inner_content():
            if isinstance(block, Table):
                for row in block.rows:
                    lines.append(" | ".join(cell.text for cell in row.cells))
            else:
                lines.append(block.text)
    except UploadError:
        raise
    except Exception:
        raise UploadError("Could not read this .docx file; it may be corrupted.")
    return [("line", i, text) for i, text in enumerate(lines, start=1)]


def extract(filename, data):
    """Returns {"type", "segments": [(kind, number, text)], "scanned"}.
    Raises UploadError for anything that can't be accepted."""
    ext = validate(filename, data)
    scanned = False
    if ext == ".pdf":
        segments, scanned = _extract_pdf(data)
    elif ext == ".docx":
        segments = _extract_docx(data)
    else:
        segments = _lines(_decode_text(data))
    return {"type": ext.lstrip("."), "segments": segments, "scanned": scanned}
