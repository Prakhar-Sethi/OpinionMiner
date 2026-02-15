"""
Database manager for OpinionMiner.
Handles all database operations including schema initialization and CRUD operations.
"""

import sqlite3
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from contextlib import contextmanager
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations for OpinionMiner."""

    def __init__(self, db_path: str = "data/opinionminer.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize schema
        self._initialize_schema()
        logger.info(f"Database initialized at {db_path}")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _initialize_schema(self):
        """Create database tables and indices if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Opinions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS opinions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    author TEXT,
                    source TEXT NOT NULL,
                    source_url TEXT UNIQUE NOT NULL,
                    topic TEXT,
                    quality_score REAL,
                    controversy_score REAL,
                    is_rage_bait INTEGER DEFAULT 0,
                    embedding BLOB,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    times_shown INTEGER DEFAULT 0
                )
            """)

            # Interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    opinion_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    time_spent INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (opinion_id) REFERENCES opinions(id)
                )
            """)

            # Bandit arms table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bandit_arms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    alpha REAL DEFAULT 1.0,
                    beta REAL DEFAULT 1.0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, topic)
                )
            """)

            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    selected_topics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indices for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_opinions_topic
                ON opinions(topic)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_opinions_source_url
                ON opinions(source_url)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_user_opinion
                ON interactions(user_id, opinion_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_bandit_arms_user_topic
                ON bandit_arms(user_id, topic)
            """)

            logger.info("Database schema initialized successfully")

    # Opinion CRUD operations

    def save_opinion(self, opinion_data: Dict[str, Any]) -> Optional[int]:
        """
        Save opinion to database with deduplication.

        Args:
            opinion_data: Dictionary containing opinion fields

        Returns:
            Opinion ID if saved, None if duplicate
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Check for duplicate by URL
                cursor.execute(
                    "SELECT id FROM opinions WHERE source_url = ?",
                    (opinion_data['source_url'],)
                )
                if cursor.fetchone():
                    logger.debug(f"Duplicate opinion: {opinion_data['source_url']}")
                    return None

                # Serialize embedding if present
                embedding_blob = None
                if 'embedding' in opinion_data and opinion_data['embedding'] is not None:
                    embedding_blob = pickle.dumps(opinion_data['embedding'])

                cursor.execute("""
                    INSERT INTO opinions (
                        title, content, author, source, source_url, topic,
                        quality_score, controversy_score, is_rage_bait, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    opinion_data.get('title'),
                    opinion_data.get('content'),
                    opinion_data.get('author'),
                    opinion_data.get('source'),
                    opinion_data.get('source_url'),
                    opinion_data.get('topic'),
                    opinion_data.get('quality_score'),
                    opinion_data.get('controversy_score'),
                    opinion_data.get('is_rage_bait', 0),
                    embedding_blob
                ))

                opinion_id = cursor.lastrowid
                logger.info(f"Saved opinion {opinion_id}: {opinion_data.get('title', '')[:50]}")
                return opinion_id

        except Exception as e:
            logger.error(f"Error saving opinion: {e}")
            return None

    def get_unprocessed_opinions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get opinions that haven't been processed yet.

        Args:
            limit: Maximum number of opinions to return

        Returns:
            List of opinion dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM opinions
                WHERE processed_at IS NULL
                ORDER BY collected_at DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def update_opinion_processing(self, opinion_id: int, processing_data: Dict[str, Any]):
        """
        Update opinion with processing results.

        Args:
            opinion_id: Opinion ID
            processing_data: Dictionary with topic, quality_score, is_rage_bait, embedding
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Serialize embedding if present
                embedding_blob = None
                if 'embedding' in processing_data and processing_data['embedding'] is not None:
                    embedding_blob = pickle.dumps(processing_data['embedding'])

                cursor.execute("""
                    UPDATE opinions
                    SET topic = ?, quality_score = ?, is_rage_bait = ?,
                        embedding = ?, processed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    processing_data.get('topic'),
                    processing_data.get('quality_score'),
                    processing_data.get('is_rage_bait', 0),
                    embedding_blob,
                    opinion_id
                ))

                logger.info(f"Updated processing data for opinion {opinion_id}")

        except Exception as e:
            logger.error(f"Error updating opinion processing: {e}")

    def get_opinions_by_topic(self, topic: str, limit: int = 50,
                             quality_threshold: float = 6.0) -> List[Dict[str, Any]]:
        """
        Get opinions for a specific topic.

        Args:
            topic: Topic name
            limit: Maximum number of opinions
            quality_threshold: Minimum quality score

        Returns:
            List of opinion dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM opinions
                WHERE topic = ?
                AND quality_score >= ?
                AND is_rage_bait = 0
                AND processed_at IS NOT NULL
                ORDER BY collected_at DESC
                LIMIT ?
            """, (topic, quality_threshold, limit))

            opinions = []
            for row in cursor.fetchall():
                opinion = dict(row)
                # Deserialize embedding
                if opinion['embedding']:
                    opinion['embedding'] = pickle.loads(opinion['embedding'])
                opinions.append(opinion)

            return opinions

    def get_opinion_by_id(self, opinion_id: int) -> Optional[Dict[str, Any]]:
        """Get a single opinion by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM opinions WHERE id = ?", (opinion_id,))
            row = cursor.fetchone()

            if row:
                opinion = dict(row)
                if opinion['embedding']:
                    opinion['embedding'] = pickle.loads(opinion['embedding'])
                return opinion
            return None

    def increment_opinion_shown(self, opinion_id: int):
        """Increment the times_shown counter for an opinion."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE opinions
                SET times_shown = times_shown + 1
                WHERE id = ?
            """, (opinion_id,))

    # Interaction operations

    def save_interaction(self, user_id: str, opinion_id: int,
                        action: str, time_spent: int = 0):
        """
        Save user interaction with an opinion.

        Args:
            user_id: User identifier
            opinion_id: Opinion ID
            action: Action type (click, skip, save)
            time_spent: Time spent in seconds
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO interactions (user_id, opinion_id, action, time_spent)
                    VALUES (?, ?, ?, ?)
                """, (user_id, opinion_id, action, time_spent))

                logger.info(f"Saved interaction: user={user_id}, opinion={opinion_id}, action={action}")

        except Exception as e:
            logger.error(f"Error saving interaction: {e}")

    def get_user_clicked_opinions(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get opinions that a user has clicked on.

        Args:
            user_id: User identifier
            limit: Maximum number to return

        Returns:
            List of opinion dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT o.* FROM opinions o
                INNER JOIN interactions i ON o.id = i.opinion_id
                WHERE i.user_id = ? AND i.action = 'click'
                ORDER BY i.created_at DESC
                LIMIT ?
            """, (user_id, limit))

            opinions = []
            for row in cursor.fetchall():
                opinion = dict(row)
                if opinion['embedding']:
                    opinion['embedding'] = pickle.loads(opinion['embedding'])
                opinions.append(opinion)

            return opinions

    def get_user_shown_opinion_ids(self, user_id: str) -> List[int]:
        """Get list of opinion IDs already shown to user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT opinion_id FROM interactions
                WHERE user_id = ?
            """, (user_id,))

            return [row[0] for row in cursor.fetchall()]

    # Bandit operations

    def get_bandit_arms(self, user_id: str) -> Dict[str, Tuple[float, float]]:
        """
        Get bandit arm parameters for a user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary mapping topic to (alpha, beta) tuple
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT topic, alpha, beta
                FROM bandit_arms
                WHERE user_id = ?
            """, (user_id,))

            return {row['topic']: (row['alpha'], row['beta'])
                   for row in cursor.fetchall()}

    def update_bandit_arm(self, user_id: str, topic: str, alpha: float, beta: float):
        """
        Update bandit arm parameters.

        Args:
            user_id: User identifier
            topic: Topic name
            alpha: Alpha parameter
            beta: Beta parameter
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO bandit_arms (user_id, topic, alpha, beta, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, topic)
                    DO UPDATE SET alpha = ?, beta = ?, updated_at = CURRENT_TIMESTAMP
                """, (user_id, topic, alpha, beta, alpha, beta))

                logger.info(f"Updated bandit arm: user={user_id}, topic={topic}, alpha={alpha}, beta={beta}")

        except Exception as e:
            logger.error(f"Error updating bandit arm: {e}")

    def initialize_user_bandit_arms(self, user_id: str, topics: List[str]):
        """
        Initialize bandit arms for a new user.

        Args:
            user_id: User identifier
            topics: List of topic names
        """
        for topic in topics:
            self.update_bandit_arm(user_id, topic, alpha=1.0, beta=1.0)

        logger.info(f"Initialized {len(topics)} bandit arms for user {user_id}")

    # User profile operations

    def create_user_profile(self, user_id: str, selected_topics: List[str]):
        """
        Create a new user profile.

        Args:
            user_id: User identifier
            selected_topics: List of initially selected topics
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_profiles (user_id, selected_topics)
                    VALUES (?, ?)
                """, (user_id, ','.join(selected_topics)))

                # Initialize bandit arms
                self.initialize_user_bandit_arms(user_id, selected_topics)

                logger.info(f"Created user profile for {user_id}")

        except Exception as e:
            logger.error(f"Error creating user profile: {e}")

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM user_profiles WHERE user_id = ?
            """, (user_id,))

            row = cursor.fetchone()
            if row:
                profile = dict(row)
                profile['selected_topics'] = profile['selected_topics'].split(',')
                return profile
            return None

    # Statistics and analytics

    def get_interaction_stats(self, user_id: str) -> Dict[str, Any]:
        """Get interaction statistics for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total interactions
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN action = 'click' THEN 1 ELSE 0 END) as clicks,
                       SUM(CASE WHEN action = 'skip' THEN 1 ELSE 0 END) as skips
                FROM interactions WHERE user_id = ?
            """, (user_id,))

            stats = dict(cursor.fetchone())
            stats['ctr'] = stats['clicks'] / stats['total'] if stats['total'] > 0 else 0

            return stats

    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM opinions")
            total_opinions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM opinions WHERE processed_at IS NOT NULL")
            processed_opinions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_profiles")
            total_users = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM interactions")
            total_interactions = cursor.fetchone()[0]

            return {
                'total_opinions': total_opinions,
                'processed_opinions': processed_opinions,
                'total_users': total_users,
                'total_interactions': total_interactions
            }


# Test/CLI interface
if __name__ == "__main__":
    # Initialize database
    db = DatabaseManager()

    # Print stats
    stats = db.get_database_stats()
    print("\n=== Database Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nDatabase initialized successfully!")
