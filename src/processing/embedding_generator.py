"""
Embedding generator for OpinionMiner.
Generates semantic embeddings for opinions using Sentence-BERT.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates semantic embeddings for opinions."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding generator.

        Args:
            model_name: Name of sentence-transformers model
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")

        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Normalized embedding vector (numpy array)
        """
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)

            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (batched for efficiency).

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding

        Returns:
            List of normalized embedding vectors
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")

            # Generate embeddings in batch
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 50
            )

            # Normalize all embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error generating embeddings batch: {e}")
            raise

    def embed_opinions(self, opinions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to a list of opinions.

        Args:
            opinions: List of opinion dictionaries

        Returns:
            List of opinions with embeddings added
        """
        if not opinions:
            return opinions

        # Extract texts (combine title and content for better embeddings)
        texts = [
            f"{opinion.get('title', '')}. {opinion.get('content', '')}"
            for opinion in opinions
        ]

        # Generate embeddings
        embeddings = self.generate_embeddings(texts)

        # Add embeddings to opinions
        for opinion, embedding in zip(opinions, embeddings):
            opinion['embedding'] = np.array(embedding)

        logger.info(f"Added embeddings to {len(opinions)} opinions")
        return opinions

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-1)
        """
        # Ensure normalized
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)

        return float(similarity)

    def compute_average_embedding(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute average of multiple embeddings.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Average embedding (normalized)
        """
        if not embeddings:
            raise ValueError("Cannot compute average of empty embedding list")

        # Stack and average
        embeddings_array = np.array(embeddings)
        avg_embedding = np.mean(embeddings_array, axis=0)

        # Normalize
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        return avg_embedding


# Test/CLI interface
if __name__ == "__main__":
    generator = EmbeddingGenerator()

    # Test with sample texts
    test_texts = [
        "Artificial intelligence will transform software development",
        "Machine learning is changing how we write code",
        "Climate change requires immediate action on carbon emissions"
    ]

    print("\n=== Embedding Generator Test ===")

    # Generate embeddings
    embeddings = generator.generate_embeddings(test_texts)

    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Test similarity
    sim_1_2 = generator.compute_similarity(
        np.array(embeddings[0]),
        np.array(embeddings[1])
    )
    sim_1_3 = generator.compute_similarity(
        np.array(embeddings[0]),
        np.array(embeddings[2])
    )

    print(f"\nSimilarity between text 1 and 2 (related): {sim_1_2:.3f}")
    print(f"Similarity between text 1 and 3 (unrelated): {sim_1_3:.3f}")
    print("\nTest passed!" if sim_1_2 > sim_1_3 else "\nTest failed!")
