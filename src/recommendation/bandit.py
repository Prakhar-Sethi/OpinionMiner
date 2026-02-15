"""
Thompson Sampling Multi-Armed Bandit for OpinionMiner.
Personalizes topic selection based on user interactions.
"""

import logging
import numpy as np
import random
from typing import Dict, Tuple, List
from src.database.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThompsonSamplingBandit:
    """Multi-armed bandit using Thompson Sampling for topic selection."""

    # Available topics (arms)
    TOPICS = [
        'technology',
        'politics',
        'philosophy',
        'science',
        'culture',
        'economics',
        'education',
        'health',
        'environment',
        'sports',
        'entertainment',
        'business',
        'other'
    ]

    def __init__(self, db_path: str = "data/opinionminer.db", epsilon: float = 0.1):
        """
        Initialize Thompson Sampling bandit.

        Args:
            db_path: Path to database
            epsilon: Exploration rate (0.0-1.0)
        """
        self.db = DatabaseManager(db_path)
        self.epsilon = epsilon  # Epsilon-greedy exploration

    def select_topic(self, user_id: str, available_topics: List[str] = None) -> str:
        """
        Select a topic for the user using Thompson Sampling.

        Args:
            user_id: User identifier
            available_topics: List of topics to choose from (defaults to all topics)

        Returns:
            Selected topic name
        """
        # Use all topics if not specified
        if available_topics is None:
            available_topics = self.TOPICS

        # Load user's bandit arms
        bandit_arms = self.db.get_bandit_arms(user_id)

        # Initialize missing arms
        for topic in available_topics:
            if topic not in bandit_arms:
                bandit_arms[topic] = (1.0, 1.0)  # Beta(1, 1) uniform prior

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            selected_topic = random.choice(available_topics)
            logger.debug(f"Exploration: randomly selected {selected_topic}")
            return selected_topic

        # Thompson Sampling exploitation
        samples = {}
        for topic in available_topics:
            alpha, beta = bandit_arms.get(topic, (1.0, 1.0))
            # Sample from Beta distribution
            sample = np.random.beta(alpha, beta)
            samples[topic] = sample

        # Select topic with highest sample
        selected_topic = max(samples, key=samples.get)
        logger.debug(f"Exploitation: selected {selected_topic} (sample={samples[selected_topic]:.3f})")

        return selected_topic

    def update(self, user_id: str, topic: str, reward: int):
        """
        Update bandit arm parameters based on user feedback.

        Args:
            user_id: User identifier
            topic: Topic that was shown
            reward: Reward value (1 for click, 0 for skip)
        """
        # Get current parameters
        bandit_arms = self.db.get_bandit_arms(user_id)
        alpha, beta = bandit_arms.get(topic, (1.0, 1.0))

        # Update parameters
        if reward == 1:
            alpha += 1  # Success: increment alpha
        else:
            beta += 1   # Failure: increment beta

        # Save to database
        self.db.update_bandit_arm(user_id, topic, alpha, beta)

        # Calculate estimated CTR for logging
        estimated_ctr = alpha / (alpha + beta)
        logger.info(f"Updated bandit arm: user={user_id}, topic={topic}, " +
                   f"reward={reward}, alpha={alpha:.2f}, beta={beta:.2f}, " +
                   f"est_ctr={estimated_ctr:.3f}")

    def get_arm_stats(self, user_id: str) -> List[Dict]:
        """
        Get statistics for all bandit arms.

        Args:
            user_id: User identifier

        Returns:
            List of arm statistics
        """
        bandit_arms = self.db.get_bandit_arms(user_id)

        stats = []
        for topic, (alpha, beta) in bandit_arms.items():
            # Calculate statistics
            estimated_ctr = alpha / (alpha + beta)
            total_pulls = (alpha + beta) - 2  # Subtract prior (1, 1)

            # Calculate confidence interval (95%)
            # Using normal approximation for Beta distribution
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            std_error = np.sqrt(variance)
            ci_lower = max(0, estimated_ctr - 1.96 * std_error)
            ci_upper = min(1, estimated_ctr + 1.96 * std_error)

            stats.append({
                'topic': topic,
                'alpha': alpha,
                'beta': beta,
                'total_pulls': int(total_pulls),
                'estimated_ctr': estimated_ctr,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })

        # Sort by estimated CTR (descending)
        stats.sort(key=lambda x: x['estimated_ctr'], reverse=True)

        return stats

    def reset_user_arms(self, user_id: str, topics: List[str] = None):
        """
        Reset bandit arms for a user.

        Args:
            user_id: User identifier
            topics: Topics to reset (defaults to all)
        """
        if topics is None:
            topics = self.TOPICS

        for topic in topics:
            self.db.update_bandit_arm(user_id, topic, alpha=1.0, beta=1.0)

        logger.info(f"Reset {len(topics)} bandit arms for user {user_id}")


# Test/CLI interface
if __name__ == "__main__":
    import sys
    import os

    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    bandit = ThompsonSamplingBandit()

    # Create test user
    test_user_id = "test_user_bandit"

    # Initialize bandit arms
    bandit.reset_user_arms(test_user_id, ['technology', 'economics', 'philosophy'])

    print("\n=== Thompson Sampling Bandit Test ===")
    print("\nInitial state (uniform priors):")
    stats = bandit.get_arm_stats(test_user_id)
    for stat in stats:
        print(f"  {stat['topic']}: alpha={stat['alpha']:.1f}, beta={stat['beta']:.1f}, " +
              f"est_ctr={stat['estimated_ctr']:.3f}")

    # Simulate interactions
    print("\nSimulating 20 interactions...")
    print("(User prefers 'technology', dislikes 'philosophy')")

    for i in range(20):
        topic = bandit.select_topic(test_user_id, ['technology', 'economics', 'philosophy'])

        # Simulate user behavior
        if topic == 'technology':
            reward = 1 if random.random() < 0.8 else 0  # 80% click rate
        elif topic == 'economics':
            reward = 1 if random.random() < 0.5 else 0  # 50% click rate
        else:  # philosophy
            reward = 1 if random.random() < 0.2 else 0  # 20% click rate

        bandit.update(test_user_id, topic, reward)

    print("\nFinal state (after learning):")
    stats = bandit.get_arm_stats(test_user_id)
    for stat in stats:
        print(f"  {stat['topic']}: alpha={stat['alpha']:.1f}, beta={stat['beta']:.1f}, " +
              f"est_ctr={stat['estimated_ctr']:.3f}, pulls={stat['total_pulls']}")

    # Test selection distribution
    print("\nTesting selection distribution (100 samples):")
    selections = {'technology': 0, 'economics': 0, 'philosophy': 0}
    for _ in range(100):
        topic = bandit.select_topic(test_user_id, ['technology', 'economics', 'philosophy'])
        selections[topic] += 1

    for topic, count in sorted(selections.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count}%")

    print("\nTest complete! Technology should be selected most often.")
