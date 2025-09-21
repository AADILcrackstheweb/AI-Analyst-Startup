# src/data/models.py

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
from pydantic import HttpUrl, EmailStr

class FundingStage(str, Enum):
    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    SERIES_D_PLUS = "series_d_plus"
    IPO = "ipo"
    ACQUIRED = "acquired"
    OTHER = "other"

class Industry(str, Enum):
    SAAS = "saas"
    ECOMMERCE = "ecommerce"
    FINTECH = "fintech"
    HEALTHTECH = "healthtech"
    EDTECH = "edtech"
    AI_ML = "ai_ml"
    BLOCKCHAIN = "blockchain"
    MARKETPLACE = "marketplace"
    CONSUMER = "consumer"
    ENTERPRISE = "enterprise"
    OTHER = "other"

class BusinessModel(str, Enum):
    SUBSCRIPTION = "subscription"
    MARKETPLACE = "marketplace"
    ECOMMERCE = "ecommerce"
    ADVERTISING = "advertising"
    FREEMIUM = "freemium"
    TRANSACTION = "transaction"
    LICENSING = "licensing"
    CONSULTING = "consulting"
    OTHER = "other"

class StartupProfile(BaseModel):
    """Complete startup profile model"""
    company_name: str = Field(..., min_length=1, max_length=200)
    industry: Industry
    stage: FundingStage
    funding_raised: Optional[float] = Field(None, ge=0, description="Total funding raised in USD")
    team_size: Optional[int] = Field(None, ge=1, le=10000)
    location: Optional[str] = Field(None, max_length=100)
    founded_year: Optional[int] = Field(None, ge=1900, le=2030)
    website: Optional[HttpUrl] = None
    
    # Business details
    product_description: str = Field(..., min_length=10, max_length=2000)
    target_market: str = Field(..., min_length=5, max_length=500)
    business_model: BusinessModel
    revenue_model: Optional[str] = Field(None, max_length=500)
    
    # Key metrics
    monthly_revenue: Optional[float] = Field(None, ge=0)
    monthly_active_users: Optional[int] = Field(None, ge=0)
    customer_acquisition_cost: Optional[float] = Field(None, ge=0)
    lifetime_value: Optional[float] = Field(None, ge=0)
    churn_rate: Optional[float] = Field(None, ge=0, le=1)
    
    # Team information
    founder_names: Optional[List[str]] = Field(None, max_items=10)
    key_team_members: Optional[List[str]] = Field(None, max_items=20)
    
    # Additional metadata
    tags: Optional[List[str]] = Field(None, max_items=20)
    notes: Optional[str] = Field(None, max_length=5000)
    
    @validator('company_name')
    def validate_company_name(cls, v):
        if not v.strip():
            raise ValueError('Company name cannot be empty')
        return v.strip()
    
    @validator('product_description', 'target_market')
    def validate_text_fields(cls, v):
        if not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    class Config:
        use_enum_values = True
        validate_assignment = True


class MarketSizeData(BaseModel):
    """Market size analysis results"""
    tam: float = Field(..., ge=0, description="Total Addressable Market in USD")
    sam: float = Field(..., ge=0, description="Serviceable Addressable Market in USD")
    som: float = Field(..., ge=0, description="Serviceable Obtainable Market in USD")
    
    # Projections
    tam_projections: Dict[str, float] = Field(default_factory=dict)
    sam_projections: Dict[str, float] = Field(default_factory=dict)
    som_projections: Dict[str, float] = Field(default_factory=dict)
    
    # Methodology and confidence
    methodology: str
    data_sources: List[str] = Field(default_factory=list)
    confidence: ConfidenceScore
    
    @validator('tam', 'sam', 'som')
    def validate_market_sizes(cls, v):
        if v < 0:
            raise ValueError('Market size cannot be negative')
        return v
    
    @root_validator
    def validate_market_hierarchy(cls, values):
        tam = values.get('tam', 0)
        sam = values.get('sam', 0)
        som = values.get('som', 0)
        
        if sam > tam:
            raise ValueError('SAM cannot be larger than TAM')
        if som > sam:
            raise ValueError('SOM cannot be larger than SAM')
        
        return values

class SentimentData(BaseModel):
    """Sentiment analysis results"""
    overall_score: float = Field(..., ge=-1, le=1, description="Overall sentiment score")
    sentiment_label: str = Field(..., description="Sentiment label: negative, neutral, positive")
    
    # Component sentiments
    social_media_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    news_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    market_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    consumer_sentiment: Optional[float] = Field(None, ge=-1, le=1)
    
    # Detailed analysis
    sentiment_sources: List[str] = Field(default_factory=list)
    sentiment_trends: Optional[Dict[str, float]] = None
    key_themes: List[str] = Field(default_factory=list)
    confidence: ConfidenceScore
    
    @validator('sentiment_label')
    def validate_sentiment_label(cls, v):
        if v not in ['negative', 'neutral', 'positive']:
            raise ValueError('Sentiment label must be negative, neutral, or positive')
        return v

class CompetitorProfile(BaseModel):
    """Individual competitor profile"""
    name: str = Field(..., min_length=1, max_length=200)
    website: Optional[HttpUrl] = None
    description: Optional[str] = Field(None, max_length=1000)
    
    # Business details
    industry: Optional[Industry] = None
    stage: Optional[FundingStage] = None
    funding_raised: Optional[float] = Field(None, ge=0)
    employee_count: Optional[int] = Field(None, ge=1)
    
    # Competitive metrics
    market_share: Optional[float] = Field(None, ge=0, le=1)
    competitive_strength: float = Field(..., ge=0, le=1)
    
    # Features and positioning
    key_features: List[str] = Field(default_factory=list)
    pricing_model: Optional[str] = None
    target_customers: Optional[str] = None
    
    # Analysis metadata
    data_sources: List[str] = Field(default_factory=list)

class CompetitiveAnalysis(BaseModel):
    """Competitive landscape analysis"""
    competitors: List[CompetitorProfile] = Field(default_factory=list)
    market_position: str = Field(..., description="Startup's position in market")
    competitive_advantages: List[str] = Field(default_factory=list)
    competitive_threats: List[str] = Field(default_factory=list)

class MarketAnalysis(BaseModel):
    """Complete market analysis results"""
    market_size: MarketSizeData
    growth_rate: float = Field(..., description="Annual market growth rate")
    market_trends: List[str] = Field(default_factory=list)
    market_opportunities: List[str] = Field(default_factory=list)
    market_threats: List[str] = Field(default_factory=list)
    
    # Analysis metadata
    data_sources: List[str] = Field(default_factory=list)


class DueDiligenceResults(BaseModel):
    """Complete due diligence analysis results"""
    # Input data
    startup_profile: StartupProfile
    documents_analyzed: List[Document]
    
    # Analysis results
    market_analysis: MarketAnalysis
    sentiment_analysis: SentimentData
    competitive_analysis: CompetitiveAnalysis
    
    # Final assessment
    due_diligence_score: float = Field(..., ge=0, le=100, description="Overall DD score out of 100")
    investment_recommendation: str = Field(..., description="Investment recommendation")
    
    # Insights and recommendations
    key_insights: List[str] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    
    # Analysis metadata
    analysis_date: datetime = Field(default_factory=datetime.now)
    analysis_duration: float = Field(..., ge=0, description="Analysis duration in seconds")
    overall_confidence: ConfidenceScore
    
    class Config:
        use_enum_values = True
        validate_assignment = True




from pydantic import BaseModel
from typing import List, Dict

class CustomerSegments(BaseModel):
    primary: List[str]
    secondary: List[str]
    details: str

class MarketSize(BaseModel):
    TAM: str
    SAM: str
    SOM: str
    assumptions: str

class CompetitiveLandscape(BaseModel):
    major_competitors: List[str]
    substitutes_or_alternatives: List[str]
    barriers_to_entry: List[str]

class MarketAnalysis(BaseModel):
    customer_segments: CustomerSegments
    market_size: MarketSize
    market_trends: List[str]
    opportunities_and_gaps: List[str]
    competitive_landscape: CompetitiveLandscape
    customer_pain_points: List[str]
    monetization_and_business_models: List[str]
    risks_and_challenges: List[str]
