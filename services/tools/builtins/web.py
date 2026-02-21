"""
Web Tools

Web search and fetch tools.
"""

from dataclasses import dataclass, field
from typing import Any
import json


@dataclass
class WebSearchResult:
    """A web search result."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    score: float = 0.0


class WebSearchTool:
    """
    Web search tool.

    Searches the web using various providers.
    """

    def __init__(
        self,
        provider: str = "stub",  # stub, serper, brave, google
        api_key: str | None = None,
    ):
        self._provider = provider
        self._api_key = api_key

    async def search(
        self,
        query: str,
        num_results: int = 5,
    ) -> list[WebSearchResult]:
        """
        Search the web.

        Args:
            query: Search query
            num_results: Number of results

        Returns:
            List of search results
        """
        if self._provider == "stub":
            return self._stub_search(query, num_results)
        elif self._provider == "serper" and self._api_key:
            return await self._serper_search(query, num_results)
        elif self._provider == "brave" and self._api_key:
            return await self._brave_search(query, num_results)
        else:
            return self._stub_search(query, num_results)

    def _stub_search(self, query: str, num_results: int) -> list[WebSearchResult]:
        """Return stub search results for testing."""
        return [
            WebSearchResult(
                title=f"Result {i+1} for: {query}",
                url=f"https://example.com/result-{i+1}",
                snippet=f"This is a sample result for the query '{query}'. It contains relevant information.",
                score=1.0 - (i * 0.1),
            )
            for i in range(num_results)
        ]

    async def _serper_search(self, query: str, num_results: int) -> list[WebSearchResult]:
        """Search using Serper API."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": num_results},
                )
                data = response.json()

                return [
                    WebSearchResult(
                        title=r.get("title", ""),
                        url=r.get("link", ""),
                        snippet=r.get("snippet", ""),
                        score=1.0 - (i * 0.1),
                    )
                    for i, r in enumerate(data.get("organic", [])[:num_results])
                ]
        except Exception:
            return self._stub_search(query, num_results)

    async def _brave_search(self, query: str, num_results: int) -> list[WebSearchResult]:
        """Search using Brave API."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": self._api_key},
                    params={"q": query, "count": num_results},
                )
                data = response.json()

                return [
                    WebSearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("description", ""),
                        score=1.0 - (i * 0.1),
                    )
                    for i, r in enumerate(data.get("web", {}).get("results", [])[:num_results])
                ]
        except Exception:
            return self._stub_search(query, num_results)


class WebFetchTool:
    """
    Web fetch tool.

    Fetches and extracts content from web pages.
    """

    def __init__(self, user_agent: str | None = None):
        self._user_agent = user_agent or "Mozilla/5.0 (compatible; AegisBot/1.0)"

    async def fetch(
        self,
        url: str,
        extract_text: bool = True,
        max_length: int = 10000,
    ) -> dict[str, Any]:
        """
        Fetch a web page.

        Args:
            url: URL to fetch
            extract_text: Extract text content
            max_length: Maximum content length

        Returns:
            Dict with url, content, title, etc.
        """
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": self._user_agent},
                    timeout=30.0,
                    follow_redirects=True,
                )

                content = response.text

                if extract_text:
                    content = self._extract_text(content)

                content = content[:max_length]

                return {
                    "url": str(response.url),
                    "status": response.status_code,
                    "content": content,
                    "title": self._extract_title(response.text),
                }
        except Exception as e:
            return {
                "url": url,
                "status": 0,
                "error": str(e),
                "content": "",
                "title": "",
            }

    def _extract_text(self, html: str) -> str:
        """Extract text from HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer"]):
                element.decompose()

            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            return "\n".join(lines)
        except ImportError:
            # Fallback: simple regex-based extraction
            import re

            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

    def _extract_title(self, html: str) -> str:
        """Extract title from HTML."""
        import re

        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
