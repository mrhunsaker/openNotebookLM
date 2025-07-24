import os
import json
import re
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Docling API Server", version="1.0.0")

class ParseRequest(BaseModel):
    text: str

class ChunkMetadata(BaseModel):
    source: str = "docling_parser"
    chunk_type: str = "paragraph"
    section: str = ""
    page: int = None

class ParsedChunk(BaseModel):
    text: str
    metadata: ChunkMetadata

class ParseResponse(BaseModel):
    chunks: List[ParsedChunk]
    total_chunks: int
    status: str = "success"

class DoclingParser:
    """
    Simple document parser that mimics Docling functionality.
    This is a basic implementation for demonstration purposes.
    """

    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200

    def parse_text(self, text: str) -> List[ParsedChunk]:
        """
        Parse text into structured chunks with metadata.
        """
        chunks = []

        # Clean and normalize text
        cleaned_text = self._clean_text(text)

        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(cleaned_text)

        # Process each paragraph
        current_section = ""
        chunk_id = 0

        for para in paragraphs:
            if not para.strip():
                continue

            # Detect if this is a heading/section
            if self._is_heading(para):
                current_section = para.strip()
                # Add heading as its own chunk
                chunks.append(ParsedChunk(
                    text=para.strip(),
                    metadata=ChunkMetadata(
                        source="docling_parser",
                        chunk_type="heading",
                        section=current_section
                    )
                ))
                chunk_id += 1
            else:
                # Split long paragraphs into smaller chunks
                para_chunks = self._chunk_paragraph(para)
                for chunk_text in para_chunks:
                    if chunk_text.strip():
                        chunks.append(ParsedChunk(
                            text=chunk_text.strip(),
                            metadata=ChunkMetadata(
                                source="docling_parser",
                                chunk_type="paragraph",
                                section=current_section
                            )
                        ))
                        chunk_id += 1

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove HTML tags if present
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double line breaks or paragraph indicators
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)

        # Also split on certain patterns that indicate paragraph breaks
        result = []
        for para in paragraphs:
            # Further split very long single paragraphs
            if len(para) > self.chunk_size * 2:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > self.chunk_size:
                        if current_chunk:
                            result.append(current_chunk)
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                if current_chunk:
                    result.append(current_chunk)
            else:
                result.append(para)

        return [p.strip() for p in result if p.strip()]

    def _is_heading(self, text: str) -> bool:
        """Detect if text is likely a heading."""
        text = text.strip()

        # Common heading patterns
        heading_patterns = [
            r'^#+\s+',  # Markdown headings
            r'^\d+\.\s+[A-Z]',  # Numbered sections
            r'^[A-Z][A-Z\s]{2,}[A-Z]$',  # ALL CAPS headings
            r'^(Chapter|Section|Part)\s+\d+',  # Chapter/Section indicators
        ]

        for pattern in heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        # Check if it's short and title-case
        if (len(text) < 100 and
            text.istitle() and
            not text.endswith('.') and
            len(text.split()) < 10):
            return True

        return False

    def _chunk_paragraph(self, paragraph: str) -> List[str]:
        """Split a paragraph into smaller chunks if needed."""
        if len(paragraph) <= self.chunk_size:
            return [paragraph]

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

# Initialize parser
parser = DoclingParser()

@app.get("/")
async def root():
    return {"message": "Docling API Server", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "docling_api"}

@app.post("/parse", response_model=ParseResponse)
async def parse_document(request: ParseRequest):
    """
    Parse document text into structured chunks.
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text content is required")

        logger.info(f"Parsing document with {len(request.text)} characters")

        # Parse the text
        chunks = parser.parse_text(request.text)

        logger.info(f"Generated {len(chunks)} chunks")

        return ParseResponse(
            chunks=chunks,
            total_chunks=len(chunks),
            status="success"
        )

    except Exception as e:
        logger.error(f"Error parsing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing document: {str(e)}")

@app.post("/parse/simple")
async def parse_document_simple(request: ParseRequest):
    """
    Simple parsing endpoint that returns chunks as a list of dictionaries.
    This matches the expected format from the main application.
    """
    try:
        if not request.text or not request.text.strip():
            return []

        chunks = parser.parse_text(request.text)

        # Convert to simple dictionary format
        result = []
        for chunk in chunks:
            result.append({
                "text": chunk.text,
                "metadata": {
                    "source": chunk.metadata.source,
                    "chunk_type": chunk.metadata.chunk_type,
                    "section": chunk.metadata.section
                }
            })

        return result

    except Exception as e:
        logger.error(f"Error in simple parsing: {str(e)}")
        # Return fallback simple chunking
        try:
            # Simple fallback: split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', request.text)
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) > 1000 and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {"source": "fallback_simple_split"}
                    })
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence

            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {"source": "fallback_simple_split"}
                })

            return chunks
        except:
            return [{
                "text": request.text,
                "metadata": {"source": "error_fallback"}
            }]

if __name__ == "__main__":
    port = int(os.getenv("DOCLING_PORT", 8001))
    logger.info(f"Starting Docling API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
