# src/tools/web_search_tool.py

import requests
import json
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import quote_plus
from pydantic import BaseModel, Field, validator
from enum import Enum

from .base_tool import BaseDueDiligenceTool
from src.data.models import ConfidenceScore

class SearchType(str, Enum):
    GENERAL = "general"
    NEWS = "news"
    ACADEMIC = "academic"
    SOCIAL = "social"
    FINANCIAL = "financial"

class WebSearchInput(BaseModel):
    """Input model for web search tool"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    search_type: SearchType = Field(default=SearchType.GENERAL, description="Type of search")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    language: str = Field(default="en", description="Search language")
    region: Optional[str] = Field(None, description="Geographic region for search")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class SearchResult(BaseModel):
    """Individual search result"""
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Result URL")
    snippet: str = Field(..., description="Result snippet/description")
    source: str = Field(..., description="Search engine source")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score")
    published_date: Optional[datetime] = Field(None, description="Publication date if available")

class WebSearchOutput(BaseModel):
    """Output model for web search tool"""
    query: str = Field(..., description="Original search query")
    search_type: SearchType = Field(..., description="Type of search performed")
    total_results: int = Field(..., ge=0, description="Total number of results")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    confidence: ConfidenceScore = Field(..., description="Search confidence score")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    data_sources: List[str] = Field(default_factory=list, description="Data sources used")

class WebSearchTool(BaseDueDiligenceTool):
    """Web search tool with Pydantic validation"""
    
    name = "web_search_tool"
    description = "Search the web for market research and competitive intelligence"
    input_model = WebSearchInput
    output_model = WebSearchOutput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.google_api_key = self.settings.GOOGLE_SEARCH_API_KEY
        self.google_cse_id = self.settings.GOOGLE_CSE_ID
        self.bing_api_key = self.settings.BING_SEARCH_API_KEY
        self.news_api_key = self.settings.NEWS_API_KEY
    
    async def _execute(self, validated_input: WebSearchInput) -> Dict[str, Any]:
        """Execute web search with validated input"""
        start_time = time.time()
        
        results = []
        data_sources = []
        
        # Perform searches based on type
        if validated_input.search_type == SearchType.GENERAL:
            if self.google_api_key and self.google_cse_id:
                google_results = await self._google_search(validated_input.query, validated_input.max_results // 2)
                results.extend(google_results)
                data_sources.append("google")
            
            if self.bing_api_key:
                bing_results = await self._bing_search(validated_input.query, validated_input.max_results // 2)
                results.extend(bing_results)
                data_sources.append("bing")
                
        elif validated_input.search_type == SearchType.NEWS:
            if self.news_api_key:
                news_results = await self._news_search(validated_input.query, validated_input.max_results)
                results.extend(news_results)
                data_sources.append("news_api")
        
        elif validated_input.search_type == SearchType.FINANCIAL:
            # Use both Google and Bing with financial-specific queries
            financial_query = f"{validated_input.query} financial market analysis"
            if self.google_api_key and self.google_cse_id:
                google_results = await self._google_search(financial_query, validated_input.max_results // 2)
                results.extend(google_results)
                data_sources.append("google_financial")
        
        # Process and validate results
        processed_results = []
        for result in results:
            try:
                # Ensure all required fields are present
                if all(key in result for key in ['title', 'url', 'snippet', 'source']):
                    search_result = SearchResult(
                        title=result['title'],
                        url=result['url'],
                        snippet=result['snippet'],
                        source=result['source'],
                        relevance_score=result.get('relevance_score', 0.5),
                        published_date=result.get('published_date')
                    )
                    processed_results.append(search_result)
            except Exception as e:
                self.logger.warning(f"Invalid search result: {e}")
                continue
        
        # Remove duplicates based on URL
        unique_results = []
        seen_urls = set()
        for result in processed_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Calculate confidence
        confidence_score = self._calculate_search_confidence(unique_results)
        confidence = ConfidenceScore(
            score=confidence_score,
            level="high" if confidence_score >= 0.7 else "medium" if confidence_score >= 0.4 else "low"
        )
        
        execution_time = time.time() - start_time
        
        return {
            "query": validated_input.query,
            "search_type": validated_input.search_type,
            "total_results": len(unique_results),
            "results": [result.dict() for result in unique_results],
            "confidence": confidence.dict(),
            "execution_time": execution_time,
            "data_sources": data_sources
        }
    
    async def _google_search(self, query: str, max_results: int) -> List[Dict]:
        """Search using Google Custom Search API"""
        if not self.google_api_key or not self.google_cse_id:
            return []
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": min(max_results, 10)
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google",
                    "relevance_score": self._calculate_relevance(item, query)
                })
            
            return results
        except Exception as e:
            self.logger.error(f"Google search failed: {str(e)}")
            return []
    
    async def _bing_search(self, query: str, max_results: int) -> List[Dict]:
        """Search using Bing Search API"""
        if not self.bing_api_key:
            return []
        
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
        params = {
            "q": query,
            "count": min(max_results, 50),
            "responseFilter": "Webpages"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("webPages", {}).get("value", []):
                results.append({
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "bing",
                    "relevance_score": self._calculate_relevance(item, query)
                })
            
            return results
        except Exception as e:
            self.logger.error(f"Bing search failed: {str(e)}")
            return []
    
    async def _news_search(self, query: str, max_results: int) -> List[Dict]:
        """Search news articles using NewsAPI"""
        if not self.news_api_key:
            return []
        
        url = "https://newsapi.org/v2/everything"
        headers = {"X-API-Key": self.news_api_key}
        params = {
            "q": query,
            "sortBy": "relevancy",
            "pageSize": min(max_results, 100),
            "language": "en"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for article in data.get("articles", []):
                published_date = None
                if article.get("publishedAt"):
                    try:
                        published_date = datetime.fromisoformat(article["publishedAt"].replace('Z', '+00:00'))
                    except:
                        pass
                
                results.append({
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "snippet": article.get("description", ""),
                    "source": f"news_{article.get('source', {}).get('name', 'unknown')}",
                    "published_date": published_date,
                    "relevance_score": self._calculate_relevance(article, query)
                })
            
            return results
        except Exception as e:
            self.logger.error(f"News search failed: {str(e)}")
            return []
    
    def _calculate_relevance(self, item: Dict, query: str) -> float:
        """Calculate relevance score for search result"""
        title = item.get("title", "").lower() if item.get("title") else ""
        snippet = item.get("snippet", "").lower() if item.get("snippet") else ""
        description = item.get("description", "").lower() if item.get("description") else ""
        
        # Combine snippet and description for news articles
        content = f"{snippet} {description}".strip()
        
        query_terms = query.lower().split()
        
        score = 0.0
        total_terms = len(query_terms)
        
        if total_terms == 0:
            return 0.0
        
        for term in query_terms:
            # Title matches are weighted more heavily
            if term in title:
                score += 2.0
            # Content matches
            if term in content:
                score += 1.0
        
        # Normalize score
        max_possible_score = total_terms * 3.0  # Max if all terms in both title and content
        normalized_score = score / max_possible_score if max_possible_score > 0 else 0.0
        
        return min(normalized_score, 1.0)
    
    def _calculate_search_confidence(self, results: List[SearchResult]) -> float:
        """Calculate confidence score for search results"""
        if not results:
            return 0.0
        
        # Factor 1: Number of results (more results = higher confidence)
        result_count_factor = min(len(results) / 10, 1.0)
        
        # Factor 2: Average relevance score
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        
        # Factor 3: Source diversity (more sources = higher confidence)
        unique_sources = len(set(r.source for r in results))
        source_diversity_factor = min(unique_sources / 3, 1.0)
        
        # Weighted combination
        confidence = (
            result_count_factor * 0.4 +
            avg_relevance * 0.4 +
            source_diversity_factor * 0.2
        )
        
        return min(confidence, 1.0)