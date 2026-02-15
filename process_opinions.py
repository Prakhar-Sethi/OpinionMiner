"""
Processing pipeline for OpinionMiner.
Processes unprocessed opinions: quality filtering, topic classification, and embedding generation.
"""

import sys
import os
import logging
from src.database.db_manager import DatabaseManager
from src.processing.quality_filter import QualityFilter
from src.processing.topic_classifier import TopicClassifier
from src.processing.embedding_generator import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_opinions(batch_size: int = 50):
    """
    Process unprocessed opinions in the database.

    Args:
        batch_size: Number of opinions to process per batch
    """
    logger.info("Starting opinion processing pipeline...")

    # Initialize components
    db = DatabaseManager()
    quality_filter = QualityFilter()
    topic_classifier = TopicClassifier()
    embedding_generator = EmbeddingGenerator()

    # Get unprocessed opinions
    unprocessed = db.get_unprocessed_opinions(limit=batch_size)

    if not unprocessed:
        logger.info("No unprocessed opinions found.")
        return

    logger.info(f"Processing {len(unprocessed)} opinions...")

    processed_count = 0
    filtered_count = 0

    for opinion in unprocessed:
        try:
            # Quality filtering
            quality_result = quality_filter.filter_opinion(
                opinion['content'],
                opinion['title']
            )

            quality_score = quality_result['quality_score']
            is_rage_bait = quality_result['is_rage_bait']

            # Check if passes threshold
            if not quality_filter.passes_threshold(quality_score, is_rage_bait):
                logger.info(f"Filtered out (low quality): {opinion['title'][:50]}... " +
                          f"(score={quality_score:.2f}, rage_bait={is_rage_bait})")
                filtered_count += 1

                # Still update database but mark as filtered
                db.update_opinion_processing(opinion['id'], {
                    'quality_score': quality_score,
                    'is_rage_bait': 1 if is_rage_bait else 0,
                    'topic': 'other',
                    'embedding': None
                })
                continue

            # Topic classification
            topic = topic_classifier.classify_opinion(
                opinion['content'],
                opinion['title']
            )

            # Embedding generation
            embedding_text = f"{opinion['title']}. {opinion['content']}"
            embedding = embedding_generator.generate_embedding(embedding_text)

            # Update database
            db.update_opinion_processing(opinion['id'], {
                'quality_score': quality_score,
                'is_rage_bait': 1 if is_rage_bait else 0,
                'topic': topic,
                'embedding': embedding
            })

            processed_count += 1
            logger.info(f"Processed: {opinion['title'][:50]}... " +
                       f"(topic={topic}, quality={quality_score:.2f})")

        except Exception as e:
            logger.error(f"Error processing opinion {opinion['id']}: {e}")
            continue

    logger.info(f"\nProcessing complete!")
    logger.info(f"  Processed successfully: {processed_count}")
    logger.info(f"  Filtered out: {filtered_count}")
    logger.info(f"  Total: {len(unprocessed)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process unprocessed opinions')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of opinions to process')

    args = parser.parse_args()

    process_opinions(batch_size=args.batch_size)
