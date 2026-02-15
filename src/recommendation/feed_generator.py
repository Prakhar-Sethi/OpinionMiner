"""
Feed generator for OpinionMiner.
Generates personalized opinion feeds using bandit and ranking algorithms.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from src.database.db_manager import DatabaseManager
from src.recommendation.bandit import ThompsonSamplingBandit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedGenerator:
    """Generates personalized opinion feeds."""

    def __init__(self, db_path: str = "data/opinionminer.db", quality_threshold: float = 6.0):
        """
        Initialize feed generator.

        Args:
            db_path: Path to database
            quality_threshold: Minimum quality score for opinions
        """
        self.db = DatabaseManager(db_path)
        self.bandit = ThompsonSamplingBandit(db_path)
        self.quality_threshold = quality_threshold

    def generate_feed(self, user_id: str, num_opinions: int = 10) -> List[Dict[str, Any]]:
        """
        Generate personalized feed for a user.

        Args:
            user_id: User identifier
            num_opinions: Number of opinions to return

        Returns:
            List of opinion dictionaries
        """
        logger.info(f"Generating feed for user {user_id} ({num_opinions} opinions)")

        # Get user profile to know their selected topics
        user_profile = self.db.get_user_profile(user_id)
        if not user_profile:
            logger.warning(f"No profile found for user {user_id}")
            return []

        available_topics = user_profile['selected_topics']

        # Select topic using bandit
        selected_topic = self.bandit.select_topic(user_id, available_topics)
        logger.info(f"Bandit selected topic: {selected_topic}")

        # Get candidate opinions from selected topic
        candidates = self.db.get_opinions_by_topic(
            topic=selected_topic,
            limit=50,
            quality_threshold=self.quality_threshold
        )

        if not candidates:
            logger.warning(f"No opinions found for topic {selected_topic}")
            # Fallback: try other topics
            for fallback_topic in available_topics:
                if fallback_topic != selected_topic:
                    candidates = self.db.get_opinions_by_topic(
                        topic=fallback_topic,
                        limit=50,
                        quality_threshold=self.quality_threshold
                    )
                    if candidates:
                        logger.info(f"Using fallback topic: {fallback_topic}")
                        break

        if not candidates:
            logger.warning("No opinions available for any topic")
            return []

        # Filter out already shown opinions
        shown_ids = set(self.db.get_user_shown_opinion_ids(user_id))
        candidates = [op for op in candidates if op['id'] not in shown_ids]

        if not candidates:
            logger.warning("All opinions already shown to user")
            # Reset shown opinions for this topic and try again
            candidates = self.db.get_opinions_by_topic(
                topic=selected_topic,
                limit=50,
                quality_threshold=self.quality_threshold
            )

        # Rank candidates
        ranked_opinions = self._rank_opinions(candidates)

        # Re-rank top candidates using embedding similarity
        top_candidates = ranked_opinions[:min(20, len(ranked_opinions))]
        reranked_opinions = self._rerank_by_similarity(user_id, top_candidates)

        # Return top N
        feed = reranked_opinions[:num_opinions]

        logger.info(f"Generated feed with {len(feed)} opinions")
        return feed

    def _rank_opinions(self, opinions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank opinions using composite scoring.

        Args:
            opinions: List of opinion dictionaries

        Returns:
            Sorted list of opinions
        """
        scored_opinions = []

        for opinion in opinions:
            # Calculate component scores
            quality_score = opinion.get('quality_score', 5.0)
            recency_score = self._calculate_recency_score(opinion.get('collected_at'))
            diversity_score = self._calculate_diversity_score(opinion)
            exploration_bonus = self._calculate_exploration_bonus(opinion)

            # Composite score
            composite_score = (
                0.4 * (quality_score / 10.0) +
                0.3 * recency_score +
                0.2 * diversity_score +
                0.1 * exploration_bonus
            )

            opinion['composite_score'] = composite_score
            scored_opinions.append(opinion)

        # Sort by composite score (descending)
        scored_opinions.sort(key=lambda x: x['composite_score'], reverse=True)

        return scored_opinions

    def _calculate_recency_score(self, collected_at) -> float:
        """
        Calculate recency score (1.0 for recent, decay exponentially).

        Args:
            collected_at: Collection timestamp

        Returns:
            Recency score (0-1)
        """
        try:
            if isinstance(collected_at, str):
                collected_dt = datetime.fromisoformat(collected_at)
            else:
                collected_dt = collected_at

            # Calculate age in days
            age_days = (datetime.now() - collected_dt).days

            # Exponential decay (half-life of 7 days)
            recency_score = np.exp(-age_days / 7.0)

            return float(recency_score)

        except Exception as e:
            logger.debug(f"Error calculating recency: {e}")
            return 0.5  # Default

    def _calculate_diversity_score(self, opinion: Dict[str, Any]) -> float:
        """
        Calculate diversity score (favor different sources/authors).

        Args:
            opinion: Opinion dictionary

        Returns:
            Diversity score (0-1)
        """
        # For now, return moderate score
        # In production, would track recent sources/authors
        return 0.7

    def _calculate_exploration_bonus(self, opinion: Dict[str, Any]) -> float:
        """
        Calculate exploration bonus (favor less-shown opinions).

        Args:
            opinion: Opinion dictionary

        Returns:
            Exploration bonus (0-1)
        """
        times_shown = opinion.get('times_shown', 0)

        # Higher bonus for less-shown opinions
        if times_shown == 0:
            return 1.0
        elif times_shown == 1:
            return 0.7
        elif times_shown == 2:
            return 0.4
        else:
            return 0.2

    def _rerank_by_similarity(self, user_id: str,
                              opinions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank opinions using embedding similarity to user's preferences.

        Args:
            user_id: User identifier
            opinions: List of opinion dictionaries

        Returns:
            Re-ranked list of opinions
        """
        # Get user's clicked opinions
        clicked_opinions = self.db.get_user_clicked_opinions(user_id, limit=20)

        if not clicked_opinions or not any(op.get('embedding') is not None for op in clicked_opinions):
            logger.debug("No clicked opinions with embeddings, skipping similarity re-ranking")
            return opinions

        # Calculate average embedding (user profile)
        clicked_embeddings = [op['embedding'] for op in clicked_opinions
                             if op.get('embedding') is not None]

        if not clicked_embeddings:
            return opinions

        user_profile_embedding = np.mean(clicked_embeddings, axis=0)
        user_profile_embedding = user_profile_embedding / np.linalg.norm(user_profile_embedding)

        # Calculate similarity scores
        for opinion in opinions:
            if opinion.get('embedding') is not None:
                similarity = np.dot(user_profile_embedding, opinion['embedding'])
                # Combine with composite score
                opinion['final_score'] = (
                    0.7 * opinion.get('composite_score', 0.5) +
                    0.3 * float(similarity)
                )
            else:
                opinion['final_score'] = opinion.get('composite_score', 0.5)

        # Sort by final score
        opinions.sort(key=lambda x: x['final_score'], reverse=True)

        return opinions


# Test/CLI interface
if __name__ == "__main__":
    import sys
    import os

    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    generator = FeedGenerator()

    # Check if we have any processed opinions
    db_stats = generator.db.get_database_stats()
    print("\n=== Feed Generator Test ===")
    print(f"Database stats:")
    print(f"  Total opinions: {db_stats['total_opinions']}")
    print(f"  Processed opinions: {db_stats['processed_opinions']}")

    if db_stats['processed_opinions'] == 0:
        print("\nNo processed opinions in database. Run data collection and processing first.")
        sys.exit(1)

    # Create test user
    test_user_id = "test_user_feed"
    test_topics = ['technology', 'economics', 'philosophy']

    # Create user profile if doesn't exist
    existing_profile = generator.db.get_user_profile(test_user_id)
    if not existing_profile:
        generator.db.create_user_profile(test_user_id, test_topics)
        print(f"\nCreated test user with topics: {', '.join(test_topics)}")
    else:
        print(f"\nUsing existing test user")

    # Generate feed
    print(f"\nGenerating feed...")
    feed = generator.generate_feed(test_user_id, num_opinions=5)

    if feed:
        print(f"\n=== Generated Feed ({len(feed)} opinions) ===")
        for i, opinion in enumerate(feed, 1):
            print(f"\n{i}. {opinion['title'][:80]}...")
            print(f"   Topic: {opinion['topic']}")
            print(f"   Quality: {opinion['quality_score']:.2f}")
            print(f"   Source: {opinion['source']}")
            if 'composite_score' in opinion:
                print(f"   Score: {opinion['composite_score']:.3f}")
    else:
        print("\nFailed to generate feed (no opinions available)")

    print("\nTest complete!")
