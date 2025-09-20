# src/tools/market_size_estimator.py

import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import aiohttp
import json
from dataclasses import dataclass

from .base_tool import BaseDueDiligenceTool
from src.data.models import MarketAnalysis, ConfidenceScore

class MarketSizeInput(BaseModel):
    """Input model for market size estimation"""
    industry: str = Field(..., min_length=1, description="Industry or market sector")
    product_category: Optional[str] = Field(None, description="Specific product category")
    target_geography: str = Field(default="Global", description="Geographic market scope")
    time_horizon: int = Field(default=5, ge=1, le=10, description="Forecast time horizon in years")
    company_stage: str = Field(default="startup", description="Company stage (startup/growth/mature)")
    include_tam_sam_som: bool = Field(default=True, description="Include TAM/SAM/SOM analysis")
    include_growth_projections: bool = Field(default=True, description="Include growth projections")
    include_market_trends: bool = Field(default=True, description="Include market trend analysis")
    
    @validator('industry')
    def validate_industry(cls, v):
        if not v.strip():
            raise ValueError('Industry cannot be empty')
        return v.strip()
    
    @validator('target_geography')
    def validate_geography(cls, v):
        valid_geographies = [
            "Global", "North America", "Europe", "Asia Pacific", "Latin America",
            "Middle East & Africa", "United States", "China", "India", "Japan"
        ]
        if v not in valid_geographies:
            raise ValueError(f'Geography must be one of: {", ".join(valid_geographies)}')
        return v

class MarketSegment(BaseModel):
    """Market segment information"""
    name: str = Field(..., description="Segment name")
    size_usd: float = Field(..., ge=0, description="Market size in USD")
    growth_rate: float = Field(..., description="Annual growth rate percentage")
    key_drivers: List[str] = Field(default_factory=list, description="Key growth drivers")
    challenges: List[str] = Field(default_factory=list, description="Market challenges")
    opportunities: List[str] = Field(default_factory=list, description="Market opportunities")

class TAMSAMSOMAnalysis(BaseModel):
    """Total Addressable Market, Serviceable Addressable Market, Serviceable Obtainable Market analysis"""
    tam_usd: float = Field(..., ge=0, description="Total Addressable Market in USD")
    sam_usd: float = Field(..., ge=0, description="Serviceable Addressable Market in USD")
    som_usd: float = Field(..., ge=0, description="Serviceable Obtainable Market in USD")
    tam_description: str = Field(..., description="TAM calculation methodology")
    sam_description: str = Field(..., description="SAM calculation methodology")
    som_description: str = Field(..., description="SOM calculation methodology")
    market_penetration_rate: float = Field(..., ge=0, le=100, description="Expected market penetration percentage")

class GrowthProjection(BaseModel):
    """Market growth projection"""
    year: int = Field(..., description="Projection year")
    market_size_usd: float = Field(..., ge=0, description="Projected market size in USD")
    growth_rate: float = Field(..., description="Year-over-year growth rate")
    key_factors: List[str] = Field(default_factory=list, description="Key factors driving growth")

class MarketTrend(BaseModel):
    """Market trend information"""
    trend_name: str = Field(..., description="Trend name")
    impact_level: str = Field(..., description="Impact level (high/medium/low)")
    time_frame: str = Field(..., description="Time frame for trend impact")
    description: str = Field(..., description="Trend description")
    market_impact_usd: Optional[float] = Field(None, ge=0, description="Estimated market impact in USD")

class CompetitiveLandscape(BaseModel):
    """Competitive landscape overview"""
    market_concentration: str = Field(..., description="Market concentration level")
    top_players: List[str] = Field(default_factory=list, description="Top market players")
    market_share_distribution: Dict[str, float] = Field(default_factory=dict, description="Market share by player")
    barriers_to_entry: List[str] = Field(default_factory=list, description="Barriers to market entry")
    competitive_intensity: str = Field(..., description="Competitive intensity level")

class MarketSizeOutput(BaseModel):
    """Output model for market size estimation"""
    industry: str = Field(..., description="Analyzed industry")
    target_geography: str = Field(..., description="Geographic scope")
    current_market_size_usd: float = Field(..., ge=0, description="Current market size in USD")
    projected_market_size_usd: float = Field(..., ge=0, description="Projected market size in USD")
    cagr_percentage: float = Field(..., description="Compound Annual Growth Rate")
    
    # Detailed analysis
    market_segments: List[MarketSegment] = Field(default_factory=list, description="Market segments")
    tam_sam_som: Optional[TAMSAMSOMAnalysis] = Field(None, description="TAM/SAM/SOM analysis")
    growth_projections: List[GrowthProjection] = Field(default_factory=list, description="Growth projections")
    market_trends: List[MarketTrend] = Field(default_factory=list, description="Market trends")
    competitive_landscape: Optional[CompetitiveLandscape] = Field(None, description="Competitive landscape")
    
    # Insights and recommendations
    key_insights: List[str] = Field(default_factory=list, description="Key market insights")
    investment_attractiveness: str = Field(..., description="Investment attractiveness rating")
    risk_factors: List[str] = Field(default_factory=list, description="Market risk factors")
    recommendations: List[str] = Field(default_factory=list, description="Strategic recommendations")
    
    # Metadata
    data_sources: List[str] = Field(default_factory=list, description="Data sources used")
    confidence: ConfidenceScore = Field(..., description="Analysis confidence")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")

class MarketSizeEstimatorTool(BaseDueDiligenceTool):
    """Market size estimation tool with comprehensive market analysis"""
    
    name = "market_size_estimator"
    description = "Estimate market size and analyze market opportunities"
    input_model = MarketSizeInput
    output_model = MarketSizeOutput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session = None
        
        # Market data cache (in production, use external databases)
        self.market_data_cache = {}
    
    async def _execute(self, validated_input: MarketSizeInput) -> Dict[str, Any]:
        """Execute market size estimation with validated input"""
        start_time = time.time()
        
        # Initialize HTTP session
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Get base market data
            market_data = await self._get_market_data(
                validated_input.industry,
                validated_input.target_geography,
                validated_input.product_category
            )
            
            # Calculate current and projected market size
            current_size, projected_size, cagr = self._calculate_market_size(
                market_data,
                validated_input.time_horizon
            )
            
            # Analyze market segments
            segments = await self._analyze_market_segments(
                validated_input.industry,
                validated_input.target_geography
            )
            
            # TAM/SAM/SOM analysis
            tam_sam_som = None
            if validated_input.include_tam_sam_som:
                tam_sam_som = await self._calculate_tam_sam_som(
                    validated_input.industry,
                    validated_input.target_geography,
                    validated_input.company_stage,
                    current_size
                )
            
            # Growth projections
            projections = []
            if validated_input.include_growth_projections:
                projections = self._generate_growth_projections(
                    current_size,
                    cagr,
                    validated_input.time_horizon
                )
            
            # Market trends analysis
            trends = []
            if validated_input.include_market_trends:
                trends = await self._analyze_market_trends(
                    validated_input.industry,
                    validated_input.target_geography
                )
            
            # Competitive landscape
            competitive_landscape = await self._analyze_competitive_landscape(
                validated_input.industry,
                validated_input.target_geography
            )
            
            # Generate insights and recommendations
            insights = self._generate_insights(
                market_data, segments, trends, competitive_landscape
            )
            
            recommendations = self._generate_recommendations(
                validated_input.industry,
                validated_input.company_stage,
                market_data,
                competitive_landscape
            )
            
            # Assess investment attractiveness
            attractiveness = self._assess_investment_attractiveness(
                cagr, competitive_landscape, trends
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(
                market_data, competitive_landscape, trends
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(market_data, segments, trends)
            
            execution_time = time.time() - start_time
            
            return {
                "industry": validated_input.industry,
                "target_geography": validated_input.target_geography,
                "current_market_size_usd": current_size,
                "projected_market_size_usd": projected_size,
                "cagr_percentage": cagr,
                "market_segments": [segment.dict() for segment in segments],
                "tam_sam_som": tam_sam_som.dict() if tam_sam_som else None,
                "growth_projections": [proj.dict() for proj in projections],
                "market_trends": [trend.dict() for trend in trends],
                "competitive_landscape": competitive_landscape.dict() if competitive_landscape else None,
                "key_insights": insights,
                "investment_attractiveness": attractiveness,
                "risk_factors": risk_factors,
                "recommendations": recommendations,
                "data_sources": ["industry_reports", "market_research", "government_data", "company_filings"],
                "confidence": confidence.dict(),
                "execution_time": execution_time
            }
    
    async def _get_market_data(
        self, 
        industry: str, 
        geography: str, 
        product_category: Optional[str]
    ) -> Dict[str, Any]:
        """Get base market data from various sources"""
        
        # Mock market data (in production, integrate with real data sources)
        cache_key = f"{industry}_{geography}_{product_category or 'general'}"
        
        if cache_key in self.market_data_cache:
            return self.market_data_cache[cache_key]
        
        # Simulate market data based on industry
        base_size = self._estimate_base_market_size(industry, geography)
        growth_rate = self._estimate_growth_rate(industry)
        
        market_data = {
            "base_market_size": base_size,
            "historical_growth_rate": growth_rate,
            "market_maturity": self._assess_market_maturity(industry),
            "regulatory_environment": self._assess_regulatory_environment(industry),
            "technology_adoption": self._assess_technology_adoption(industry),
            "economic_sensitivity": self._assess_economic_sensitivity(industry)
        }
        
        self.market_data_cache[cache_key] = market_data
        return market_data
    
    def _estimate_base_market_size(self, industry: str, geography: str) -> float:
        """Estimate base market size based on industry and geography"""
        
        # Industry size multipliers (mock data)
        industry_multipliers = {
            "software": 500_000_000_000,  # $500B
            "healthcare": 400_000_000_000,  # $400B
            "fintech": 300_000_000_000,  # $300B
            "ecommerce": 600_000_000_000,  # $600B
            "education": 200_000_000_000,  # $200B
            "manufacturing": 800_000_000_000,  # $800B
            "retail": 700_000_000_000,  # $700B
            "automotive": 900_000_000_000,  # $900B
            "energy": 1_000_000_000_000,  # $1T
            "telecommunications": 450_000_000_000  # $450B
        }
        
        # Geography multipliers
        geography_multipliers = {
            "Global": 1.0,
            "North America": 0.35,
            "Europe": 0.25,
            "Asia Pacific": 0.30,
            "United States": 0.25,
            "China": 0.15,
            "India": 0.08,
            "Japan": 0.06
        }
        
        # Find closest industry match
        industry_lower = industry.lower()
        base_size = 100_000_000_000  # Default $100B
        
        for key, size in industry_multipliers.items():
            if key in industry_lower:
                base_size = size
                break
        
        # Apply geography multiplier
        geo_multiplier = geography_multipliers.get(geography, 0.1)
        
        return base_size * geo_multiplier
    
    def _estimate_growth_rate(self, industry: str) -> float:
        """Estimate growth rate based on industry"""
        
        growth_rates = {
            "software": 12.5,
            "healthcare": 8.5,
            "fintech": 15.2,
            "ecommerce": 14.7,
            "education": 9.8,
            "manufacturing": 4.2,
            "retail": 6.5,
            "automotive": 5.8,
            "energy": 3.5,
            "telecommunications": 7.2
        }
        
        industry_lower = industry.lower()
        for key, rate in growth_rates.items():
            if key in industry_lower:
                return rate
        
        return 8.0  # Default growth rate
    
    def _calculate_market_size(
        self, 
        market_data: Dict[str, Any], 
        time_horizon: int
    ) -> tuple[float, float, float]:
        """Calculate current and projected market size"""
        
        current_size = market_data["base_market_size"]
        growth_rate = market_data["historical_growth_rate"]
        
        # Calculate CAGR (with some adjustments for market maturity)
        maturity_factor = {
            "emerging": 1.2,
            "growth": 1.0,
            "mature": 0.8,
            "declining": 0.5
        }
        
        adjusted_growth = growth_rate * maturity_factor.get(
            market_data["market_maturity"], 1.0
        )
        
        # Calculate projected size
        projected_size = current_size * ((1 + adjusted_growth / 100) ** time_horizon)
        
        return current_size, projected_size, adjusted_growth
    
    async def _analyze_market_segments(
        self, 
        industry: str, 
        geography: str
    ) -> List[MarketSegment]:
        """Analyze market segments"""
        
        # Mock segment