"""
Blog/Substack RSS scraper for OpinionMiner.
Scrapes contrarian writers and thinkers - completely free!
"""

import feedparser
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlogRSSScraper:
    """Scrapes opinions from blogs and Substacks using RSS feeds."""

    # Curated list of contrarian/thoughtful writers
    BLOG_FEEDS = {
        # Economics/Policy
        'econlog': 'https://www.econlib.org/feed/',
        'marginal_revolution': 'https://marginalrevolution.com/feed',
        'overcoming_bias': 'https://www.overcomingbias.com/feed',

        # Rationality/Philosophy
        'astral_codex_ten': 'https://astralcodexten.substack.com/feed',
        'scott_aaronson': 'https://scottaaronson.blog/feed/',
        'scholars_stage': 'https://scholars-stage.org/feed/',

        # Politics/Culture
        'freddie_deboer': 'https://freddiedeboer.substack.com/feed',
        'richard_hanania': 'https://richardhanania.substack.com/feed',
        'the_free_press': 'https://www.thefp.com/feed',

        # Tech/Science
        'construction_physics': 'https://www.construction-physics.com/feed',
        'roots_of_progress': 'https://rootsofprogress.org/feed',
    }

    def __init__(self):
        """Initialize blog RSS scraper."""
        self.min_content_length = 200  # Blogs tend to have longer content

    def scrape_feed(self, feed_name: str, feed_url: str) -> List[Dict[str, Any]]:
        """
        Scrape a single blog RSS feed.

        Args:
            feed_name: Identifier for the blog
            feed_url: RSS feed URL

        Returns:
            List of opinion dictionaries
        """
        opinions = []

        try:
            logger.info(f"Scraping {feed_name} ({feed_url})")
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                logger.warning(f"Feed parsing issue for {feed_name}: {feed.bozo_exception}")

            for entry in feed.entries:
                try:
                    opinion = self._parse_entry(entry, feed_name)
                    if opinion:
                        opinions.append(opinion)
                except Exception as e:
                    logger.error(f"Error parsing entry from {feed_name}: {e}")
                    continue

            logger.info(f"Scraped {len(opinions)} opinions from {feed_name}")

        except Exception as e:
            logger.error(f"Error scraping {feed_name}: {e}")

        return opinions

    def _parse_entry(self, entry: Any, feed_name: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single RSS entry into an opinion dictionary.

        Args:
            entry: Feedparser entry object
            feed_name: Source feed name

        Returns:
            Opinion dictionary or None if invalid
        """
        # Extract basic fields
        title = entry.get('title', '').strip()
        url = entry.get('link', '').strip()

        # Extract author (may be in different fields)
        author = entry.get('author', '')
        if not author and 'authors' in entry and entry.authors:
            author = entry.authors[0].get('name', '')
        if not author:
            author = feed_name  # Use feed name as fallback

        # Extract content - try multiple fields
        content = ''

        # Try content field (most detailed)
        if 'content' in entry and entry.content:
            content = entry.content[0].value
        # Try summary_detail
        elif 'summary_detail' in entry:
            content = entry.summary_detail.value
        # Try summary
        elif 'summary' in entry:
            content = entry.summary
        # Try description
        elif 'description' in entry:
            content = entry.description

        # Parse HTML to get clean text
        if content:
            soup = BeautifulSoup(content, 'html.parser')

            # Remove script and style tags
            for script in soup(['script', 'style']):
                script.decompose()

            # Get text content
            content_text = soup.get_text(separator=' ', strip=True)

            # Clean up whitespace
            content_text = re.sub(r'\s+', ' ', content_text).strip()
        else:
            content_text = ''

        # Skip if content too short
        if len(content_text) < self.min_content_length:
            logger.debug(f"Skipping short content from {feed_name}: {title[:50]}")
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

        # Blog posts are assumed to have moderate controversy
        # (they're thoughtful pieces, not rage bait)
        controversy_score = 0.6

        return {
            'title': title,
            'content': content_text,
            'author': author.strip(),
            'source': f'blog_{feed_name}',
            'source_url': url,
            'controversy_score': controversy_score,
            'collected_at': published_dt
        }

    def scrape_all(self) -> List[Dict[str, Any]]:
        """
        Scrape all configured blog feeds.

        Returns:
            List of all opinions from all blogs
        """
        all_opinions = []

        for feed_name, feed_url in self.BLOG_FEEDS.items():
            try:
                opinions = self.scrape_feed(feed_name, feed_url)
                all_opinions.extend(opinions)
            except Exception as e:
                logger.error(f"Failed to scrape {feed_name}: {e}")
                continue

        logger.info(f"Total opinions scraped from blogs: {len(all_opinions)}")
        return all_opinions


# Test/CLI interface
if __name__ == "__main__":
    scraper = BlogRSSScraper()
    opinions = scraper.scrape_all()

    print(f"\n=== Blog RSS Scraping Results ===")
    print(f"Total opinions: {len(opinions)}")

    if opinions:
        print(f"\nSample opinion:")
        sample = opinions[0]
        print(f"Title: {sample['title'][:80]}...")
        print(f"Author: {sample['author']}")
        print(f"Source: {sample['source']}")
        print(f"Content length: {len(sample['content'])} chars")
