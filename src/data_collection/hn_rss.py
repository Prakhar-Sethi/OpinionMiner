"""
HackerNews RSS scraper for OpinionMiner.
Uses HNRSS service - completely free, no API key required!
"""

import feedparser
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HackerNewsRSSScraper:
    """Scrapes opinions from HackerNews using HNRSS service."""

    # HNRSS feeds for controversial/debated content
    HN_FEEDS = [
        'https://hnrss.org/newest?comments>50',  # Highly debated stories
        'https://hnrss.org/newest?points>100',   # Popular stories
    ]

    def __init__(self):
        """Initialize HackerNews RSS scraper."""
        self.min_comments = 20  # Minimum comments for opinion extraction

    def scrape_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """
        Scrape a single HN RSS feed.

        Args:
            feed_url: HNRSS feed URL

        Returns:
            List of opinion dictionaries
        """
        opinions = []

        try:
            logger.info(f"Scraping HN feed: {feed_url}")
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                logger.warning(f"Feed parsing issue: {feed.bozo_exception}")

            for entry in feed.entries:
                try:
                    opinion = self._parse_entry(entry)
                    if opinion:
                        opinions.append(opinion)
                except Exception as e:
                    logger.error(f"Error parsing HN entry: {e}")
                    continue

            logger.info(f"Scraped {len(opinions)} opinions from HN")

        except Exception as e:
            logger.error(f"Error scraping HN feed: {e}")

        return opinions

    def _parse_entry(self, entry: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a single HN RSS entry.

        Args:
            entry: Feedparser entry object

        Returns:
            Opinion dictionary or None if invalid
        """
        # Extract basic fields
        title = entry.get('title', '').strip()
        url = entry.get('link', '').strip()
        comments_url = entry.get('comments', '').strip()

        # Extract metadata from summary
        summary = entry.get('summary', '')
        soup = BeautifulSoup(summary, 'html.parser')
        summary_text = soup.get_text()

        # Try to extract points and comments count
        import re
        points_match = re.search(r'Points:\s*(\d+)', summary_text)
        comments_match = re.search(r'Comments:\s*(\d+)', summary_text)

        points = int(points_match.group(1)) if points_match else 0
        num_comments = int(comments_match.group(1)) if comments_match else 0

        # Skip if not enough discussion
        if num_comments < self.min_comments:
            logger.debug(f"Skipping low-comment story: {title[:50]}")
            return None

        # For HN, we use the title + top comments as content
        # Try to fetch top comment if we have comments_url
        content = self._fetch_top_comments(comments_url) if comments_url else title

        if not content or len(content) < 100:
            content = f"{title}\n\nHackerNews discussion with {num_comments} comments."

        # Extract published date
        published = entry.get('published', None)
        if published:
            try:
                published_dt = datetime(*entry.published_parsed[:6])
            except:
                published_dt = datetime.now()
        else:
            published_dt = datetime.now()

        # Calculate controversy (high comments/points ratio = controversial)
        if points > 0:
            controversy_score = min(1.0, (num_comments / points) * 0.5)
        else:
            controversy_score = 0.7

        return {
            'title': title,
            'content': content,
            'author': 'hackernews',
            'source': 'hackernews',
            'source_url': comments_url or url,
            'controversy_score': controversy_score,
            'collected_at': published_dt
        }

    def _fetch_top_comments(self, comments_url: str, max_comments: int = 3) -> str:
        """
        Fetch top comments from HN story page.

        Args:
            comments_url: URL to HN comments page
            max_comments: Maximum comments to fetch

        Returns:
            Combined text of top comments
        """
        try:
            # Set user agent to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; OpinionMiner/1.0)'
            }

            response = requests.get(comments_url, headers=headers, timeout=5)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find comment elements (HN uses class 'comment')
            comments = []
            comment_elements = soup.find_all('div', class_='comment', limit=max_comments)

            for elem in comment_elements:
                # Extract comment text
                comment_text = elem.get_text(separator=' ', strip=True)
                if comment_text and len(comment_text) > 50:
                    comments.append(comment_text[:500])  # Limit length

            if comments:
                return '\n\n'.join(comments)

        except Exception as e:
            logger.debug(f"Could not fetch comments: {e}")

        return ''

    def scrape_all(self) -> List[Dict[str, Any]]:
        """
        Scrape all configured HN feeds.

        Returns:
            List of all opinions from HackerNews
        """
        all_opinions = []

        for feed_url in self.HN_FEEDS:
            try:
                opinions = self.scrape_feed(feed_url)
                all_opinions.extend(opinions)
            except Exception as e:
                logger.error(f"Failed to scrape HN feed: {e}")
                continue

        logger.info(f"Total opinions scraped from HackerNews: {len(all_opinions)}")
        return all_opinions


# Test/CLI interface
if __name__ == "__main__":
    scraper = HackerNewsRSSScraper()
    opinions = scraper.scrape_all()

    print(f"\n=== HackerNews RSS Scraping Results ===")
    print(f"Total opinions: {len(opinions)}")

    if opinions:
        print(f"\nSample opinion:")
        sample = opinions[0]
        print(f"Title: {sample['title'][:80]}...")
        print(f"Source: {sample['source']}")
        print(f"Content length: {len(sample['content'])} chars")
        print(f"Controversy score: {sample['controversy_score']:.2f}")
