"""
News ingestion script for NexusSignal.
Fetches news from Finnhub and NewsAPI, stores raw data in JSONL format,
and inserts recent headlines into Postgres.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import httpx
from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import insert

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.config import settings, TICKERS
from backend.app.db import get_session_maker, NewsLive


# Data paths
NEWS_DIR = Path(__file__).parent.parent / "data" / "raw" / "news"
NEWS_DIR.mkdir(parents=True, exist_ok=True)

# API URLs
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
NEWSAPI_BASE_URL = "https://newsapi.org/v2"


class NewsFetcher:
    """Fetches news from multiple sources and stores in database."""

    def __init__(self, finnhub_key: str, newsapi_key: str):
        self.finnhub_key = finnhub_key
        self.newsapi_key = newsapi_key
        self.seen_urls: set = set()

    async def fetch_finnhub_company_news(
        self,
        ticker: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Fetch company-specific news from Finnhub.

        Args:
            ticker: Stock ticker symbol
            from_date: Start date (default: 7 days ago)
            to_date: End date (default: today)

        Returns:
            List of news articles
        """
        if from_date is None:
            from_date = datetime.now() - timedelta(days=7)
        if to_date is None:
            to_date = datetime.now()

        url = f"{FINNHUB_BASE_URL}/company-news"
        params = {
            "symbol": ticker,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "token": self.finnhub_key,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                articles = response.json()

            if not articles:
                return []

            # Format articles
            formatted = []
            for article in articles:
                formatted.append({
                    "ticker": ticker,
                    "timestamp": datetime.fromtimestamp(article.get("datetime", 0)),
                    "headline": article.get("headline", ""),
                    "summary": article.get("summary", ""),
                    "source": article.get("source", "Finnhub"),
                    "url": article.get("url", ""),
                    "sentiment_score": None,  # Will be added in Phase 3
                })

            print(f"Finnhub: Fetched {len(formatted)} articles for {ticker}")
            return formatted

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"Finnhub: Rate limit hit for {ticker}")
                await asyncio.sleep(10)
            else:
                print(f"Finnhub HTTP error for {ticker}: {e.response.status_code}")
            return []
        except Exception as e:
            print(f"Finnhub error for {ticker}: {e}")
            return []

    async def fetch_newsapi_everything(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        language: str = "en",
    ) -> List[Dict]:
        """
        Fetch news articles from NewsAPI.

        Args:
            query: Search query (e.g., ticker or company name)
            from_date: Start date (default: 7 days ago)
            language: Language code (default: en)

        Returns:
            List of news articles
        """
        if from_date is None:
            from_date = datetime.now() - timedelta(days=7)

        url = f"{NEWSAPI_BASE_URL}/everything"
        params = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%d"),
            "language": language,
            "sortBy": "publishedAt",
            "apiKey": self.newsapi_key,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            articles = data.get("articles", [])
            if not articles:
                return []

            # Format articles
            formatted = []
            for article in articles:
                published_at = article.get("publishedAt", "")
                try:
                    timestamp = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                except Exception:
                    timestamp = datetime.now()

                formatted.append({
                    "ticker": None,  # NewsAPI doesn't tag by ticker
                    "timestamp": timestamp,
                    "headline": article.get("title", ""),
                    "summary": article.get("description", ""),
                    "source": article.get("source", {}).get("name", "NewsAPI"),
                    "url": article.get("url", ""),
                    "sentiment_score": None,
                })

            print(f"NewsAPI: Fetched {len(formatted)} articles for query '{query}'")
            return formatted

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"NewsAPI: Rate limit hit for query '{query}'")
                await asyncio.sleep(10)
            else:
                print(f"NewsAPI HTTP error for '{query}': {e.response.status_code}")
            return []
        except Exception as e:
            print(f"NewsAPI error for '{query}': {e}")
            return []

    def deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Remove duplicate articles by URL.

        Args:
            articles: List of article dictionaries

        Returns:
            Deduplicated list
        """
        unique = []
        for article in articles:
            url = article.get("url", "")
            if url and url not in self.seen_urls:
                self.seen_urls.add(url)
                unique.append(article)

        return unique

    async def save_to_jsonl(self, articles: List[Dict], date_str: str) -> None:
        """
        Save articles to JSONL file.

        Args:
            articles: List of article dictionaries
            date_str: Date string for filename (YYYY-MM-DD)
        """
        if not articles:
            return

        file_path = NEWS_DIR / f"{date_str}.jsonl"

        # Append to existing file or create new
        with open(file_path, "a", encoding="utf-8") as f:
            for article in articles:
                # Convert timestamp to string for JSON serialization
                article_copy = article.copy()
                article_copy["timestamp"] = article_copy["timestamp"].isoformat()
                f.write(json.dumps(article_copy) + "\n")

        print(f"Saved {len(articles)} articles to {file_path}")

    async def save_to_postgres(self, articles: List[Dict]) -> None:
        """
        Save recent articles to Postgres news_live table.

        Args:
            articles: List of article dictionaries
        """
        if not articles:
            return

        # Filter to last 24 hours only
        # Use UTC to ensure timezone compatibility
        from datetime import timezone
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        # Handle both timezone-aware and timezone-naive timestamps
        recent = []
        for a in articles:
            ts = a["timestamp"]
            # Convert timezone-naive to UTC
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                recent.append(a)

        if not recent:
            return

        try:
            session_maker = get_session_maker()

            async with session_maker() as session:
                for article in recent:
                    # Prepare record
                    record = {
                        "ticker": article.get("ticker"),
                        "timestamp": article["timestamp"],
                        "headline": article["headline"],
                        "sentiment_score": article.get("sentiment_score"),
                        "source": article["source"],
                        "url": article.get("url"),
                    }

                    # Insert with conflict handling (skip duplicates by URL)
                    stmt = insert(NewsLive).values(**record)
                    stmt = stmt.on_conflict_do_nothing(index_elements=["url"])

                    await session.execute(stmt)

                await session.commit()

            print(f"Inserted {len(recent)} articles into Postgres")

        except Exception as e:
            print(f"Error saving to Postgres: {e}")

    async def fetch_all_news(self) -> List[Dict]:
        """
        Fetch news from all sources for all tickers.

        Returns:
            Combined list of all articles
        """
        all_articles = []

        # Fetch from Finnhub for each ticker
        if self.finnhub_key:
            for ticker in TICKERS:
                articles = await self.fetch_finnhub_company_news(ticker)
                all_articles.extend(articles)
                await asyncio.sleep(1)  # Rate limiting

        # Fetch from NewsAPI for general market news
        if self.newsapi_key:
            # Search for market-wide news
            queries = ["stock market", "S&P 500", "NASDAQ", "Wall Street"]
            for query in queries:
                articles = await self.fetch_newsapi_everything(query)
                all_articles.extend(articles)
                await asyncio.sleep(2)  # Rate limiting

        return all_articles


async def main() -> None:
    """Main entry point for news ingestion."""
    # Load environment variables
    load_dotenv()

    # Check API keys
    if not settings.FINNHUB_API_KEY and not settings.NEWSAPI_API_KEY:
        print("ERROR: No news API keys found in environment variables")
        print("Please set FINNHUB_API_KEY and/or NEWSAPI_API_KEY in .env file")
        return

    print("Starting news ingestion")
    print(f"Timestamp: {datetime.now()}")
    print(f"Data directory: {NEWS_DIR}")
    print(f"Tickers: {len(TICKERS)}")

    fetcher = NewsFetcher(
        finnhub_key=settings.FINNHUB_API_KEY,
        newsapi_key=settings.NEWSAPI_API_KEY,
    )

    # Fetch all news
    print("\n" + "=" * 50)
    print("Fetching news from sources...")
    print("=" * 50)

    all_articles = await fetcher.fetch_all_news()

    # Deduplicate
    print(f"\nTotal articles fetched: {len(all_articles)}")
    unique_articles = fetcher.deduplicate_articles(all_articles)
    print(f"Unique articles: {len(unique_articles)}")

    if not unique_articles:
        print("No new articles to save")
        return

    # Save to JSONL
    date_str = datetime.now().strftime("%Y-%m-%d")
    await fetcher.save_to_jsonl(unique_articles, date_str)

    # Save to Postgres
    await fetcher.save_to_postgres(unique_articles)

    print("\n" + "=" * 50)
    print("News ingestion completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
