import time
import random
from typing import List, Dict, Any

from pydantic import BaseModel, Field, validator
from enum import Enum
from langchain_community.tools import DuckDuckGoSearchRun

class DuckDuckGoSearchType(str, Enum):
    GENERAL = "general"

class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class DuckDuckGoSearchResult(BaseModel):
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Result URL")
    snippet: str = Field(..., description="Result snippet/description")
    source: str = Field(default="duckduckgo", description="Search engine source")
    # relevance_score: float = Field(default=0.5, ge=0, le=1, description="Relevance score")

class DuckDuckGoSearchOutput(BaseModel):
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., ge=0, description="Total number of results")
    results: List[DuckDuckGoSearchResult] = Field(default_factory=list, description="Search results")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    data_sources: List[str] = Field(default_factory=lambda: ["duckduckgo"], description="Data sources used")

class DuckDuckGoSearchTool:
    """DuckDuckGo web search tool with Pydantic validation"""
    name = "duckduckgo_search_tool"
    description = "Search the web using DuckDuckGo for market research and intelligence"
    input_model = DuckDuckGoSearchInput
    output_model = DuckDuckGoSearchOutput

    def __init__(self):
        self.search_tool = DuckDuckGoSearchRun()

    def execute(self, validated_input: DuckDuckGoSearchInput) -> Dict[str, Any]:
        start_time = time.time()
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        try:
            results_raw = self.search_tool.run(validated_input.query)
            # DuckDuckGoSearchRun returns a string, so we split into results heuristically
            results = self._parse_results(results_raw, validated_input.max_results)
        except Exception as e:
            results = []
        processed_results = [DuckDuckGoSearchResult(**r) for r in results]
        execution_time = time.time() - start_time
        return {
            "query": validated_input.query,
            "total_results": len(processed_results),
            "results": [r.dict() for r in processed_results],
            "execution_time": execution_time,
            "data_sources": ["duckduckgo"]
        }

    def _parse_results(self, results_raw: str, max_results: int) -> List[Dict[str, Any]]:
        # Heuristic: split by double newlines, expect each result as 'title\nurl\nsnippet'
        results = []
        for block in results_raw.strip().split("\n\n"):
            lines = block.strip().split("\n")
            if len(lines) >= 2:
                title = lines[0]
                url = lines[1]
                snippet = lines[2] if len(lines) > 2 else ""
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "duckduckgo"
                })
            if len(results) >= max_results:
                break
        return results
