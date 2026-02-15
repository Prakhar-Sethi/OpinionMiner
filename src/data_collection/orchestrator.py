"""
Data collection orchestrator for OpinionMiner.
Coordinates scraping from all sources and saves to database.
"""

import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_collection.reddit_rss import RedditRSSScraper
from src.data_collection.rss_scraper import BlogRSSScraper
from src.data_collection.hn_rss import HackerNewsRSSScraper
from src.database.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollectionOrchestrator:
    """Orchestrates data collection from all sources."""

    def __init__(self, db_path: str = "data/opinionminer.db", include_hn: bool = False):
        """
        Initialize orchestrator.

        Args:
            db_path: Path to database
            include_hn: Whether to include HackerNews (optional)
        """
        self.db = DatabaseManager(db_path)
        self.include_hn = include_hn

        # Initialize scrapers
        self.reddit_scraper = RedditRSSScraper()
        self.blog_scraper = BlogRSSScraper()
        self.hn_scraper = HackerNewsRSSScraper() if include_hn else None

    def run_collection(self) -> Dict[str, Any]:
        """
        Run complete data collection pipeline.

        Returns:
            Statistics dictionary with collection results
        """
        logger.info("Starting data collection...")

        stats = {
            'reddit': {'collected': 0, 'saved': 0, 'duplicates': 0, 'errors': 0},
            'blogs': {'collected': 0, 'saved': 0, 'duplicates': 0, 'errors': 0},
            'hackernews': {'collected': 0, 'saved': 0, 'duplicates': 0, 'errors': 0},
            'total': {'collected': 0, 'saved': 0, 'duplicates': 0, 'errors': 0}
        }

        # Collect from Reddit RSS
        try:
            logger.info("Collecting from Reddit RSS feeds...")
            reddit_opinions = self.reddit_scraper.scrape_all()
            stats['reddit']['collected'] = len(reddit_opinions)

            # Save to database
            for opinion in reddit_opinions:
                opinion_id = self.db.save_opinion(opinion)
                if opinion_id:
                    stats['reddit']['saved'] += 1
                else:
                    stats['reddit']['duplicates'] += 1

        except Exception as e:
            logger.error(f"Error collecting from Reddit: {e}")
            stats['reddit']['errors'] += 1

        # Collect from blog RSS
        try:
            logger.info("Collecting from blog RSS feeds...")
            blog_opinions = self.blog_scraper.scrape_all()
            stats['blogs']['collected'] = len(blog_opinions)

            # Save to database
            for opinion in blog_opinions:
                opinion_id = self.db.save_opinion(opinion)
                if opinion_id:
                    stats['blogs']['saved'] += 1
                else:
                    stats['blogs']['duplicates'] += 1

        except Exception as e:
            logger.error(f"Error collecting from blogs: {e}")
            stats['blogs']['errors'] += 1

        # Optionally collect from HackerNews
        if self.include_hn and self.hn_scraper:
            try:
                logger.info("Collecting from HackerNews RSS...")
                hn_opinions = self.hn_scraper.scrape_all()
                stats['hackernews']['collected'] = len(hn_opinions)

                # Save to database
                for opinion in hn_opinions:
                    opinion_id = self.db.save_opinion(opinion)
                    if opinion_id:
                        stats['hackernews']['saved'] += 1
                    else:
                        stats['hackernews']['duplicates'] += 1

            except Exception as e:
                logger.error(f"Error collecting from HackerNews: {e}")
                stats['hackernews']['errors'] += 1

        # Calculate totals
        for source in ['reddit', 'blogs', 'hackernews']:
            for metric in ['collected', 'saved', 'duplicates', 'errors']:
                stats['total'][metric] += stats[source][metric]

        logger.info("Data collection complete!")
        self._print_stats(stats)

        return stats

    def _print_stats(self, stats: Dict[str, Any]):
        """Print collection statistics."""
        print("\n" + "="*60)
        print("DATA COLLECTION STATISTICS")
        print("="*60)

        for source in ['reddit', 'blogs', 'hackernews']:
            if stats[source]['collected'] > 0 or stats[source]['errors'] > 0:
                print(f"\n{source.upper()}:")
                print(f"  Collected:  {stats[source]['collected']}")
                print(f"  Saved:      {stats[source]['saved']}")
                print(f"  Duplicates: {stats[source]['duplicates']}")
                if stats[source]['errors'] > 0:
                    print(f"  Errors:     {stats[source]['errors']}")

        print(f"\nTOTAL:")
        print(f"  Collected:  {stats['total']['collected']}")
        print(f"  Saved:      {stats['total']['saved']}")
        print(f"  Duplicates: {stats['total']['duplicates']}")
        if stats['total']['errors'] > 0:
            print(f"  Errors:     {stats['total']['errors']}")

        print("="*60)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Collect opinions from all sources')
    parser.add_argument('--include-hn', action='store_true',
                       help='Include HackerNews (optional)')
    parser.add_argument('--db-path', type=str, default='data/opinionminer.db',
                       help='Path to database file')

    args = parser.parse_args()

    # Run collection
    orchestrator = DataCollectionOrchestrator(
        db_path=args.db_path,
        include_hn=args.include_hn
    )

    stats = orchestrator.run_collection()

    # Exit with success
    sys.exit(0)
