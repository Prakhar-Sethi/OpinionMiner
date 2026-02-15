"""
Reddit RSS scraper for OpinionMiner.
Uses free Reddit RSS feeds - no API key required!
"""

import feedparser
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditRSSScraper:
    """Scrapes opinions from Reddit using RSS feeds (completely free!)."""

    # Subreddits with contrarian/debatable content
    SUBREDDIT_FEEDS = [
        "https://www.reddit.com/r/unpopularopinion/.rss",
        "https://www.reddit.com/r/changemyview/.rss",
        "https://www.reddit.com/r/The10thDentist/.rss",
        "https://www.reddit.com/r/TrueUnpopularOpinion/.rss",
        "https://www.reddit.com/r/slatestarcodex/.rss",
        "https://www.reddit.com/r/themotte/.rss",
    ]

    def __init__(self):
        """Initialize Reddit RSS scraper."""
        self.min_content_length = 100  # Minimum characters for substantial content

    def scrape_subreddit(self, rss_url: str) -> List[Dict[str, Any]]:
        """
        Scrape a single subreddit RSS feed.

        Args:
            rss_url: RSS feed URL

        Returns:
            List of opinion dictionaries
        """
        opinions = []

        try:
            logger.info(f"Scraping {rss_url}")
            feed = feedparser.parse(rss_url)

            if feed.bozo:
                logger.warning(f"Feed parsing issue for {rss_url}: {feed.bozo_exception}")

            for entry in feed.entries:
                try:
                    opinion = self._parse_entry(entry, rss_url)
                    if opinion:
                        opinions.append(opinion)
                except Exception as e:
                    logger.error(f"Error parsing entry: {e}")
                    continue

            logger.info(f"Scraped {len(opinions)} opinions from {rss_url}")

        except Exception as e:
            logger.error(f"Error scraping {rss_url}: {e}")

        return opinions

    def _parse_entry(self, entry: Any, feed_url: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single RSS entry into an opinion dictionary.

        Args:
            entry: Feedparser entry object
            feed_url: Source feed URL

        Returns:
            Opinion dictionary or None if invalid
        """
        # Extract basic fields
        title = entry.get('title', '').strip()
        url = entry.get('link', '').strip()
        author = entry.get('author', 'unknown').strip()

        # Extract content (Reddit RSS includes HTML content)
        content = ''
        if 'content' in entry and entry.content:
            content = entry.content[0].value
        elif 'summary' in entry:
            content = entry.summary

        # Parse HTML to get text content
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            # Get text, removing HTML tags
            content_text = soup.get_text(separator=' ', strip=True)
        else:
            content_text = ''

        # Skip if content too short
        if len(content_text) < self.min_content_length:
            logger.debug(f"Skipping short content: {title[:50]}")
            return None

        # Extract published date
        published = entry.get('published', None)
        if published:
            try:
                published_dt = datetime(*entry.published_parsed[:6])
            except:
                published_dt = datetime.now()
        else:
            published_dt = datetime.now()

        # Try to extract score and comments from HTML content
        score, num_comments = self._extract_metrics(content)

        # Calculate controversy score (approximation without upvote_ratio)
        # Higher comments relative to score = more controversial
        if score > 0:
            controversy_score = (num_comments / max(score, 1)) * 0.8
        else:
            controversy_score = 0.5  # Default for unknown

        # Extract subreddit name from feed URL
        subreddit_match = re.search(r'/r/([^/]+)/', feed_url)
        subreddit = subreddit_match.group(1) if subreddit_match else 'reddit'

        return {
            'title': title,
            'content': content_text,
            'author': author,
            'source': f'reddit_{subreddit}',
            'source_url': url,
            'controversy_score': controversy_score,
            'collected_at': published_dt
        }

    def _extract_metrics(self, html_content: str) -> tuple:
        """
        Extract score and comment count from HTML content.

        Args:
            html_content: HTML content string

        Returns:
            Tuple of (score, num_comments)
        """
        score = 0
        num_comments = 0

        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Look for score patterns (varies by feed format)
            text = soup.get_text()

            # Try to find "X points" or "X score"
            score_match = re.search(r'(\d+)\s+point', text, re.IGNORECASE)
            if score_match:
                score = int(score_match.group(1))

            # Try to find "X comments"
            comments_match = re.search(r'(\d+)\s+comment', text, re.IGNORECASE)
            if comments_match:
                num_comments = int(comments_match.group(1))

        except Exception as e:
            logger.debug(f"Could not extract metrics: {e}")

        return score, num_comments

    def scrape_all(self) -> List[Dict[str, Any]]:
        """
        Scrape all configured subreddit feeds.

        Returns:
            List of all opinions from all subreddits
        """
        all_opinions = []

        for feed_url in self.SUBREDDIT_FEEDS:
            try:
                opinions = self.scrape_subreddit(feed_url)
                all_opinions.extend(opinions)
            except Exception as e:
                logger.error(f"Failed to scrape {feed_url}: {e}")
                continue

        logger.info(f"Total opinions scraped from Reddit: {len(all_opinions)}")
        return all_opinions


# Test/CLI interface
if __name__ == "__main__":
    scraper = RedditRSSScraper()
    opinions = scraper.scrape_all()

    print(f"\n=== Reddit RSS Scraping Results ===")
    print(f"Total opinions: {len(opinions)}")

    if opinions:
        print(f"\nSample opinion:")
        sample = opinions[0]
        print(f"Title: {sample['title'][:80]}...")
        print(f"Source: {sample['source']}")
        print(f"Content length: {len(sample['content'])} chars")
        print(f"Controversy score: {sample['controversy_score']:.2f}")
