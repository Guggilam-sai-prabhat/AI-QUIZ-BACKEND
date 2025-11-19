"""
Text Chunker Service
Splits long text into manageable chunks for embeddings generation.
Supports token-based chunking with tiktoken or word-based fallback.
"""

from typing import List, Optional
import re

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Falling back to word-based chunking.")


class TextChunker:
    """Handles text chunking with token-based or word-based strategies."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the text chunker.
        
        Args:
            encoding_name: The tiktoken encoding to use (default: cl100k_base for GPT-4/embeddings)
        """
        self.encoding_name = encoding_name
        self.tokenizer = None
        
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                print(f"Warning: Could not load tiktoken encoding '{encoding_name}': {e}")
                print("Falling back to word-based chunking.")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate tokens as words
            return len(text.split())
    
    def chunk_by_tokens(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Chunk text using tiktoken tokenizer.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position, accounting for overlap
            start += chunk_size - overlap
            
        return chunks
    
    def chunk_by_words(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Chunk text using word-based approach (fallback method).
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum words per chunk
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of text chunks
        """
        # Split by whitespace while preserving some structure
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            # Move start position, accounting for overlap
            start += chunk_size - overlap
            
        return chunks


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Split text into chunks with specified size and overlap.
    
    Args:
        text: Input text to be chunked
        chunk_size: Target size for each chunk in tokens (default: 500)
        overlap: Number of overlapping tokens between consecutive chunks (default: 50)
        encoding_name: Tiktoken encoding name (default: cl100k_base)
        
    Returns:
        List of text chunks
        
    Example:
        >>> text = "Your long document text here..."
        >>> chunks = chunk_text(text, chunk_size=500, overlap=50)
        >>> print(f"Created {len(chunks)} chunks")
    """
    if not text or not text.strip():
        return []
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Initialize chunker
    chunker = TextChunker(encoding_name=encoding_name)
    
    # Use appropriate chunking method
    if chunker.tokenizer:
        chunks = chunker.chunk_by_tokens(text, chunk_size, overlap)
    else:
        chunks = chunker.chunk_by_words(text, chunk_size, overlap)
    
    # Filter out empty chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    return chunks


def get_chunk_stats(text: str, chunks: List[str], encoding_name: str = "cl100k_base") -> dict:
    """
    Get statistics about the chunking operation.
    
    Args:
        text: Original text
        chunks: List of chunks
        encoding_name: Tiktoken encoding name
        
    Returns:
        Dictionary with chunking statistics
    """
    chunker = TextChunker(encoding_name=encoding_name)
    
    chunk_tokens = [chunker.count_tokens(chunk) for chunk in chunks]
    
    return {
        "num_chunks": len(chunks),
        "original_tokens": chunker.count_tokens(text),
        "total_chunk_tokens": sum(chunk_tokens),
        "avg_chunk_tokens": sum(chunk_tokens) / len(chunks) if chunks else 0,
        "min_chunk_tokens": min(chunk_tokens) if chunk_tokens else 0,
        "max_chunk_tokens": max(chunk_tokens) if chunk_tokens else 0,
    }


# Example usage and testing
if __name__ == "__main__":
    # Sample text for testing
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to 
    natural intelligence displayed by animals including humans. AI research has been defined 
    as the field of study of intelligent agents, which refers to any system that perceives 
    its environment and takes actions that maximize its chance of achieving its goals.
    
    The term "artificial intelligence" had previously been used to describe machines that 
    mimic and display "human" cognitive skills that are associated with the human mind, 
    such as "learning" and "problem-solving". This definition has since been rejected by 
    major AI researchers who now describe AI in terms of rationality and acting rationally, 
    which does not limit how intelligence can be articulated.
    
    AI applications include advanced web search engines, recommendation systems, understanding 
    human speech, self-driving cars, generative or creative tools, automated decision-making, 
    and competing at the highest level in strategic game systems.
    
    As machines become increasingly capable, tasks considered to require "intelligence" are 
    often removed from the definition of AI, a phenomenon known as the AI effect. For instance, 
    optical character recognition is frequently excluded from things considered to be AI, having 
    become a routine technology.
    """ * 10  # Repeat to create a longer text
    
    print("=" * 70)
    print("TEXT CHUNKER TEST")
    print("=" * 70)
    
    # Test with default parameters
    print("\n1. Testing with default parameters (500 tokens, 50 overlap):")
    chunks = chunk_text(sample_text)
    print(f"   Created {len(chunks)} chunks")
    
    # Get detailed stats
    stats = get_chunk_stats(sample_text, chunks)
    print(f"\n   Statistics:")
    print(f"   - Original tokens: {stats['original_tokens']}")
    print(f"   - Total chunk tokens: {stats['total_chunk_tokens']}")
    print(f"   - Average chunk size: {stats['avg_chunk_tokens']:.1f} tokens")
    print(f"   - Min chunk size: {stats['min_chunk_tokens']} tokens")
    print(f"   - Max chunk size: {stats['max_chunk_tokens']} tokens")
    
    # Show first chunk preview
    print(f"\n   First chunk preview (first 150 chars):")
    print(f"   \"{chunks[0][:150]}...\"")
    
    # Test with different parameters
    print("\n2. Testing with smaller chunks (200 tokens, 20 overlap):")
    chunks_small = chunk_text(sample_text, chunk_size=200, overlap=20)
    print(f"   Created {len(chunks_small)} chunks")
    
    # Test with very short text
    print("\n3. Testing with short text:")
    short_text = "This is a short text that should result in a single chunk."
    chunks_short = chunk_text(short_text)
    print(f"   Created {len(chunks_short)} chunk(s)")
    
    # Test with empty text
    print("\n4. Testing with empty text:")
    chunks_empty = chunk_text("")
    print(f"   Created {len(chunks_empty)} chunks")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)