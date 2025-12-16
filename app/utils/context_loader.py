"""
MCP Context Loader
Pure utility functions for formatting document chunks into LLM-ready context
"""
import re
from typing import List, Dict, Any


def _normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text while preserving paragraph structure
    
    Args:
        text: Raw text that may contain irregular whitespace
    
    Returns:
        Text with normalized whitespace
    """
    if not text:
        return ""
    
    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace 3+ newlines with double newline (preserve paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def _format_chunk(chunk: Dict[str, Any], index: int) -> str:
    """
    Format a single chunk with header and normalized content
    
    Args:
        chunk: Chunk dictionary with 'text' and optional metadata
        index: Display index (1-based) for the chunk
    
    Returns:
        Formatted chunk string
    """
    # Extract text content
    text = chunk.get("text", chunk.get("content", ""))
    normalized_text = _normalize_whitespace(text)
    
    # Build metadata line if available
    metadata_parts = []
    
    if "page" in chunk or "page_number" in chunk:
        page = chunk.get("page") or chunk.get("page_number")
        metadata_parts.append(f"Page {page}")
    
    if "score" in chunk:
        metadata_parts.append(f"Score: {chunk['score']:.3f}")
    
    # Format header
    header = f"[Chunk {index}]"
    if metadata_parts:
        header += f" ({', '.join(metadata_parts)})"
    
    return f"{header}\n{normalized_text}"


async def load_context(material_id: str, top_k_chunks: List[Dict[str, Any]]) -> str:
    """
    Load and format chunks into a clean context string for LLM consumption
    
    This is a pure function that formats pre-retrieved chunks without
    making any database calls.
    
    Args:
        material_id: The material/document ID (used for context header)
        top_k_chunks: List of chunk dictionaries from Qdrant, each containing:
            - text (or content): The chunk text content
            - chunk_id (optional): Original chunk identifier
            - page/page_number (optional): Source page number
            - score (optional): Relevance score from vector search
    
    Returns:
        A formatted context string with:
            - Document header
            - Numbered chunks with separators
            - Normalized whitespace
            - Optional metadata per chunk
    
    Example output:
        === Document Context: 507f1f77bcf86cd799439011 ===
        
        [Chunk 1] (Page 3, Score: 0.892)
        This is the content of the first chunk...
        
        ---
        
        [Chunk 2] (Page 5, Score: 0.847)
        This is the content of the second chunk...
        
        === End of Context ===
    """
    if not top_k_chunks:
        return f"=== Document Context: {material_id} ===\n\nNo chunks available.\n\n=== End of Context ==="
    
    # Sort chunks by chunk_id if available, otherwise maintain order
    sorted_chunks = sorted(
        top_k_chunks,
        key=lambda c: c.get("chunk_id", c.get("id", 0))
    )
    
    # Build context parts
    parts = []
    
    # Header
    parts.append(f"=== Document Context: {material_id} ===")
    parts.append("")  # Empty line after header
    
    # Format each chunk
    separator = "\n---\n"
    formatted_chunks = []
    
    for idx, chunk in enumerate(sorted_chunks, start=1):
        formatted = _format_chunk(chunk, idx)
        formatted_chunks.append(formatted)
    
    parts.append(separator.join(formatted_chunks))
    
    # Footer
    parts.append("")  # Empty line before footer
    parts.append(f"=== End of Context ({len(sorted_chunks)} chunks) ===")
    
    return "\n".join(parts)


async def load_multi_material_context(
    materials: List[Dict[str, Any]]
) -> str:
    """
    Load context from multiple materials into a single formatted string
    
    Args:
        materials: List of dicts, each containing:
            - material_id: The document ID
            - chunks: List of chunks for that material
    
    Returns:
        Combined formatted context string
    """
    if not materials:
        return "No materials provided."
    
    context_parts = []
    
    for material in materials:
        material_id = material.get("material_id", "unknown")
        chunks = material.get("chunks", [])
        
        material_context = await load_context(material_id, chunks)
        context_parts.append(material_context)
    
    return "\n\n".join(context_parts)