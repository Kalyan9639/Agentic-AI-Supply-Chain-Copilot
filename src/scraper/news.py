"""Scraper module using Crawl4AI."""

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "CRAWL4_AI_BASE_DIRECTORY", str(PROJECT_ROOT / "data" / "cache")
)
from models.schema import NewsArticle
from config import settings
from utils.logging import get_logger

logger = get_logger(__name__)


def _stable_article_id(source_url: str, title: str, content: str) -> str:
    """Create a deterministic article ID so duplicate detection survives restarts."""

    payload = f"{source_url}|{title}|{content}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"news_{digest[:16]}"


class NewsScraper:
    """Scrapes news articles from ITLN websites."""

    def __init__(self):
        self.urls = settings.target_urls
        self.logger = logger

    async def fetch_all(self) -> List[NewsArticle]:
        """Fetch news from all configured URLs."""
        all_articles = []

        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler(verbose=False) as crawler:
            for url in self.urls:
                try:
                    self.logger.info(f"Scraping: {url}")
                    result = await crawler.arun(url)

                    if result.success and result.markdown:
                        articles = self._parse_markdown(result.markdown, url)
                        all_articles.extend(articles)
                        self.logger.info(f"Parsed {len(articles)} articles from {url}")

                except Exception as e:
                    self.logger.error(f"Error scraping {url}: {e}")

        return all_articles

    def _parse_markdown(self, markdown: str, source_url: str) -> List[NewsArticle]:
        """
        Parse markdown content into NewsArticle objects.
        This is a basic parser - may need adjustment based on actual page structure.
        """
        articles = []

        # Split by potential article separators (heuristic)
        # In production, this would use DOM structure or better parsing
        sections = markdown.split("\n\n")

        for i, section in enumerate(sections):
            section = section.strip()
            if not section or len(section) < 20:
                continue

            # Extract title (first line) and content
            lines = section.split("\n", 1)
            title = lines[0].strip() if lines else "No Title"
            content = lines[1].strip() if len(lines) > 1 else ""

            if len(content) < 10:
                continue  # Skip too-short entries

            article = NewsArticle(
                id=_stable_article_id(source_url, title, content),
                title=title[:200],
                content=content[:2000],
                url=source_url,
                published_at=None,  # Could parse from content
                source="itln.in",
            )
            articles.append(article)

        return articles

    async def fetch_single(self, url: str) -> NewsArticle | None:
        """Fetch a single page and return as article."""
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler(verbose=False) as crawler:
            try:
                result = await crawler.arun(url)
                if result.success and result.markdown:
                    articles = self._parse_markdown(result.markdown, url)
                    return articles[0] if articles else None
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {e}")
        return None
