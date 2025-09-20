# src/tools/sentiment_analysis_tool.py (Complete version)

import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
import aiohttp
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from google.cloud import language_v1

from .base_tool import BaseDueDiligenceTool
from src.data.models import SentimentData, ConfidenceScore, SentimentScore

class SentimentAnalysisInput(BaseModel):
    """Input model for sentiment analysis"""
    texts: List[str] = Field(..., min_items=1, description="Texts to analyze for sentiment")
    sources: List[str] = Field(default_factory=list, description="Source identifiers for each text")
    include_social_media: bool = Field(default=True, description="Include social media sentiment")
    include_news: bool = Field(default=True, description="Include news sentiment")
    company_name: Optional[str] = Field(None, description="Company name for targeted analysis")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v or any(not text.strip() for text in v):
            raise ValueError('All texts must be non-empty')
        return v
    
    @validator('sources')
    def validate_sources_length(cls, v, values):
        if 'texts' in values and v and len(v) != len(values['texts']):
            raise ValueError('Sources list must match texts list length')
        return v

class SocialMediaSentiment(BaseModel):
    """Social media sentiment data"""
    platform: str = Field(..., description="Social media platform")
    mentions: int = Field(..., ge=0, description="Number of mentions")
    positive_ratio: float = Field(..., ge=0, le=1, description="Ratio of positive mentions")
    negative_ratio: float = Field(..., ge=0, le=1, description="Ratio of negative mentions")
    neutral_ratio: float = Field(..., ge=0, le=1, description="Ratio of neutral mentions")
    engagement_score: float = Field(..., ge=0, description="Engagement score")
    trending_topics: List[str] = Field(default_factory=list, description="Related trending topics")

class NewsSentiment(BaseModel):
    """News sentiment data"""
    source: str = Field(..., description="News source")
    headline: str = Field(..., description="News headline")
    sentiment_score: float = Field(..., ge=-1, le=1, description="Sentiment score")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance to company")
    publication_date: datetime = Field(..., description="Publication date")
    url: Optional[str] = Field(None, description="Article URL")

class SentimentAnalysisOutput(BaseModel):
    """Output model for sentiment analysis"""
    overall_sentiment: SentimentScore = Field(..., description="Overall sentiment analysis")
    text_sentiments: List[SentimentData] = Field(default_factory=list, description="Individual text sentiments")
    social_media_sentiment: List[SocialMediaSentiment] = Field(default_factory=list, description="Social media sentiment")
    news_sentiment: List[NewsSentiment] = Field(default_factory=list, description="News sentiment analysis")
    sentiment_trends: Dict[str, float] = Field(default_factory=dict, description="Sentiment trends over time")
    key_themes: List[str] = Field(default_factory=list, description="Key sentiment themes")
    confidence: ConfidenceScore = Field(..., description="Analysis confidence")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")

class SentimentAnalysisTool(BaseDueDiligenceTool):
    """Sentiment analysis tool with multiple providers and social media integration"""
    
    name = "sentiment_analysis_tool"
    description = "Analyze sentiment from various text sources and social media"
    input_model = SentimentAnalysisInput
    output_model = SentimentAnalysisOutput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize Google Cloud Natural Language client
        try:
            self.language_client = language_v1.LanguageServiceClient()
        except Exception:
            self.language_client = None
            self.logger.warning("Google Cloud Language client not available")
    
    async def _execute(self, validated_input: SentimentAnalysisInput) -> Dict[str, Any]:
        """Execute sentiment analysis with validated input"""
        start_time = time.time()
        
        # Analyze individual texts
        text_sentiments = await self._analyze_text_sentiments(
            validated_input.texts, 
            validated_input.sources
        )
        
        # Analyze social media sentiment if requested
        social_media_sentiment = []
        if validated_input.include_social_media and validated_input.company_name:
            social_media_sentiment = await self._analyze_social_media_sentiment(
                validated_input.company_name
            )
        
        # Analyze news sentiment if requested
        news_sentiment = []
        if validated_input.include_news and validated_input.company_name:
            news_sentiment = await self._analyze_news_sentiment(
                validated_input.company_name
            )
        
        # Calculate overall sentiment
        overall_sentiment = self._calculate_overall_sentiment(
            text_sentiments, social_media_sentiment, news_sentiment
        )
        
        # Extract key themes
        key_themes = self._extract_key_themes(validated_input.texts)
        
        # Calculate sentiment trends (simplified)
        sentiment_trends = self._calculate_sentiment_trends(
            text_sentiments, news_sentiment
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            text_sentiments, social_media_sentiment, news_sentiment
        )
        
        execution_time = time.time() - start_time
        
        return {
            "overall_sentiment": overall_sentiment.dict(),
            "text_sentiments": [sentiment.dict() for sentiment in text_sentiments],
            "social_media_sentiment": [sentiment.dict() for sentiment in social_media_sentiment],
            "news_sentiment": [sentiment.dict() for sentiment in news_sentiment],
            "sentiment_trends": sentiment_trends,
            "key_themes": key_themes,
            "confidence": confidence.dict(),
            "execution_time": execution_time
        }
    
    def _calculate_confidence(
        self,
        text_sentiments: List[SentimentData],
        social_sentiments: List[SocialMediaSentiment],
        news_sentiments: List[NewsSentiment]
    ) -> ConfidenceScore:
        """Calculate confidence in sentiment analysis"""
        
        # Base confidence on data availability and consistency
        data_sources = 0
        if text_sentiments:
            data_sources += 1
        if social_sentiments:
            data_sources += 1
        if news_sentiments:
            data_sources += 1
        
        # Calculate consistency (variance in sentiment scores)
        all_scores = []
        if text_sentiments:
            all_scores.extend([s.sentiment_score for s in text_sentiments])
        if news_sentiments:
            all_scores.extend([s.sentiment_score for s in news_sentiments])
        
        # Calculate variance
        if len(all_scores) > 1:
            mean_score = sum(all_scores) / len(all_scores)
            variance = sum((score - mean_score) ** 2 for score in all_scores) / len(all_scores)
            consistency = max(0, 1 - variance)
        else:
            consistency = 0.5
        
        # Combine factors
        confidence_score = (data_sources / 3) * 0.6 + consistency * 0.4
        
        if confidence_score >= 0.7:
            level = "high"
        elif confidence_score >= 0.4:
            level = "medium"
        else:
            level = "low"
        
        return ConfidenceScore(score=confidence_score, level=level)