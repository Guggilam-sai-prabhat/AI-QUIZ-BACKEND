"""
Embedding Service
Generates vector embeddings for text chunks using SentenceTransformers or OpenAI.
"""

from typing import List, Optional, Dict
import os
from functools import lru_cache

# Try importing SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

# Try importing OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: pip install openai")


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def __init__(self):
        self.model_name = None
        self.dimension = None
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        raise NotImplementedError
    
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.dimension


class SentenceTransformerProvider(EmbeddingProvider):
    """SentenceTransformers embedding provider (local models)."""
    
    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
        "all-MiniLM-L12-v2": 384,
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize SentenceTransformer model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        super().__init__()
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.dimension = self.MODEL_DIMENSIONS.get(model_name, 384)
        
        print(f"✓ Loaded SentenceTransformer model: {model_name} (dim: {self.dimension})")
    
    @staticmethod
    @lru_cache(maxsize=3)
    def _load_model(model_name: str) -> SentenceTransformer:
        """Load and cache the model."""
        return SentenceTransformer(model_name)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using SentenceTransformers.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Convert to list of lists
        return embeddings.tolist()


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key (or set OPENAI_API_KEY environment variable)
        """
        super().__init__()
        
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai is not installed. "
                "Install with: pip install openai"
            )
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)
        self.dimension = self.MODEL_DIMENSIONS.get(model_name, 1536)
        
        print(f"✓ Initialized OpenAI client with model: {model_name} (dim: {self.dimension})")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # OpenAI API has a batch limit, process in batches if needed
        batch_size = 2048
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Call OpenAI API
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            
            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings


def get_embeddings(
    chunks: List[str],
    model: str = "sentence-transformers",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> List[List[float]]:
    """
    Generate embeddings for text chunks.
    
    Args:
        chunks: List of text chunks to embed
        model: Provider type - "sentence-transformers" or "openai"
        model_name: Specific model name (optional)
            - For sentence-transformers: defaults to "all-MiniLM-L6-v2"
            - For openai: defaults to "text-embedding-3-small"
        api_key: API key for OpenAI (optional, can use OPENAI_API_KEY env var)
        
    Returns:
        List of embedding vectors matching chunk order
        
    Example:
        >>> chunks = ["Hello world", "How are you?"]
        >>> embeddings = get_embeddings(chunks)
        >>> print(f"Generated {len(embeddings)} embeddings")
    """
    if not chunks:
        return []
    
    # Initialize the appropriate provider
    if model.lower() == "sentence-transformers":
        model_name = model_name or "all-MiniLM-L6-v2"
        provider = SentenceTransformerProvider(model_name=model_name)
    elif model.lower() == "openai":
        model_name = model_name or "text-embedding-3-small"
        provider = OpenAIProvider(model_name=model_name, api_key=api_key)
    else:
        raise ValueError(
            f"Unknown model type: {model}. "
            "Use 'sentence-transformers' or 'openai'"
        )
    
    # Generate embeddings
    embeddings = provider.embed(chunks)
    
    return embeddings


def get_embedding_dimension(
    model: str = "sentence-transformers",
    model_name: Optional[str] = None
) -> int:
    """
    Get the embedding dimension for a specific model.
    
    Args:
        model: Provider type - "sentence-transformers" or "openai"
        model_name: Specific model name (optional)
        
    Returns:
        Embedding dimension
    """
    if model.lower() == "sentence-transformers":
        model_name = model_name or "all-MiniLM-L6-v2"
        return SentenceTransformerProvider.MODEL_DIMENSIONS.get(model_name, 384)
    elif model.lower() == "openai":
        model_name = model_name or "text-embedding-3-small"
        return OpenAIProvider.MODEL_DIMENSIONS.get(model_name, 1536)
    else:
        raise ValueError(f"Unknown model type: {model}")


def get_available_models() -> Dict[str, List[str]]:
    """
    Get a list of available models for each provider.
    
    Returns:
        Dictionary mapping provider names to available models
    """
    return {
        "sentence-transformers": list(SentenceTransformerProvider.MODEL_DIMENSIONS.keys()),
        "openai": list(OpenAIProvider.MODEL_DIMENSIONS.keys())
    }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("EMBEDDING SERVICE TEST")
    print("=" * 70)
    
    # Sample text chunks
    sample_chunks = [
        "Artificial intelligence is transforming the world.",
        "Machine learning models can process vast amounts of data.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Vector embeddings represent text as numerical vectors."
    ]
    
    print(f"\nTest data: {len(sample_chunks)} text chunks")
    print(f"First chunk: \"{sample_chunks[0]}\"")
    
    # Test 1: SentenceTransformers (default)
    print("\n" + "-" * 70)
    print("TEST 1: SentenceTransformers (Local Model)")
    print("-" * 70)
    
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            embeddings_st = get_embeddings(
                sample_chunks,
                model="sentence-transformers"
            )
            
            print(f"✓ Generated {len(embeddings_st)} embeddings")
            print(f"  Embedding dimension: {len(embeddings_st[0])}")
            print(f"  First embedding (first 5 values): {embeddings_st[0][:5]}")
            print(f"  Data type: {type(embeddings_st[0][0])}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("✗ SentenceTransformers not available")
        print("  Install with: pip install sentence-transformers")
    
    # Test 2: Different SentenceTransformer model
    print("\n" + "-" * 70)
    print("TEST 2: SentenceTransformers (Different Model)")
    print("-" * 70)
    
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            embeddings_st2 = get_embeddings(
                sample_chunks[:2],  # Just test with 2 chunks
                model="sentence-transformers",
                model_name="paraphrase-MiniLM-L6-v2"
            )
            
            print(f"✓ Generated {len(embeddings_st2)} embeddings")
            print(f"  Embedding dimension: {len(embeddings_st2[0])}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("✗ SentenceTransformers not available")
    
    # Test 3: OpenAI (only if API key available)
    print("\n" + "-" * 70)
    print("TEST 3: OpenAI Embeddings")
    print("-" * 70)
    
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            embeddings_openai = get_embeddings(
                sample_chunks[:2],  # Just test with 2 chunks to save API costs
                model="openai",
                model_name="text-embedding-3-small"
            )
            
            print(f"✓ Generated {len(embeddings_openai)} embeddings")
            print(f"  Embedding dimension: {len(embeddings_openai[0])}")
            print(f"  First embedding (first 5 values): {embeddings_openai[0][:5]}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        if not OPENAI_AVAILABLE:
            print("✗ OpenAI not available")
            print("  Install with: pip install openai")
        else:
            print("✗ OpenAI API key not found")
            print("  Set OPENAI_API_KEY environment variable to test")
    
    # Test 4: Get available models
    print("\n" + "-" * 70)
    print("TEST 4: Available Models")
    print("-" * 70)
    
    available = get_available_models()
    print("\nSentenceTransformers models:")
    for model in available["sentence-transformers"]:
        dim = get_embedding_dimension("sentence-transformers", model)
        print(f"  - {model} (dim: {dim})")
    
    print("\nOpenAI models:")
    for model in available["openai"]:
        dim = get_embedding_dimension("openai", model)
        print(f"  - {model} (dim: {dim})")
    
    # Test 5: Edge cases
    print("\n" + "-" * 70)
    print("TEST 5: Edge Cases")
    print("-" * 70)
    
    # Empty list
    empty_embeddings = get_embeddings([])
    print(f"Empty list: {len(empty_embeddings)} embeddings")
    
    # Single chunk
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        single_embedding = get_embeddings(["Single text chunk"])
        print(f"Single chunk: {len(single_embedding)} embedding(s)")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    # Print summary
    print("\nSUMMARY:")
    print(f"  SentenceTransformers available: {SENTENCE_TRANSFORMERS_AVAILABLE}")
    print(f"  OpenAI available: {OPENAI_AVAILABLE}")
    print(f"  OpenAI API key configured: {bool(os.getenv('OPENAI_API_KEY'))}")