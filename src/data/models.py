# src/data/models.py

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import HttpUrl, EmailStr

class BaseResponse(BaseModel):
    """Base response model with common fields"""
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True
    message: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConfidenceScore(BaseModel):
    """Model for confidence scoring"""
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    level: str = Field(..., description="Confidence level: low, medium, high")
    
    @validator('level')
    def validate_level(cls, v):
        if v not in ['low', 'medium', 'high']:
            raise ValueError('Level must be low, medium, or high')
        return v
    
    @root_validator
    def validate_score_level_consistency(cls, values):
        score = values.get('score', 0)
        level = values.get('level', '')
        
        if score < 0.3 and level != 'low':
            raise ValueError('Score < 0.3 must have level "low"')
        elif 0.3 <= score < 0.7 and level != 'medium':
            raise ValueError('Score 0.3-0.7 must have level "medium"')
        elif score >= 0.7 and level != 'high':
            raise ValueError('Score >= 0.7 must have level "high"')
        
        return values

class FundingStage(str, Enum):
    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    SERIES_D_PLUS = "series_d_plus"
    IPO = "ipo"
    ACQUIRED = "acquired"

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

class DocumentType(str, Enum):
    PITCH_DECK = "pitch_deck"
    TRANSCRIPT = "transcript"
    MEETING_NOTES = "meeting_notes"
    FINANCIAL_STATEMENT = "financial_statement"
    BUSINESS_PLAN = "business_plan"
    MARKET_RESEARCH = "market_research"
    LEGAL_DOCUMENT = "legal_document"
    OTHER = "other"

class DocumentSource(str, Enum):
    UPLOAD = "upload"
    URL = "url"
    GCS = "gcs"
    EMAIL = "email"

class Document(BaseModel):
    """Document model for processing"""
    id: Optional[str] = Field(None, description="Unique document identifier")
    name: str = Field(..., min_length=1, max_length=255)
    type: DocumentType
    source: DocumentSource
    
    # Source-specific fields
    file_path: Optional[str] = None
    url: Optional[HttpUrl] = None
    gcs_path: Optional[str] = None
    
    # Metadata
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    mime_type: Optional[str] = None
    upload_date: datetime = Field(default_factory=datetime.now)
    
    # Processing status
    processed: bool = False
    processing_error: Optional[str] = None
    
    @root_validator
    def validate_source_fields(cls, values):
        source = values.get('source')
        file_path = values.get('file_path')
        url = values.get('url')
        gcs_path = values.get('gcs_path')
        
        if source == DocumentSource.UPLOAD and not file_path:
            raise ValueError('file_path required for upload source')
        elif source == DocumentSource.URL and not url:
            raise ValueError('url required for url source')
        elif source == DocumentSource.GCS and not gcs_path:
            raise ValueError('gcs_path required for gcs source')
        
        return values

class ProcessedDocument(BaseModel):
    """Processed document with extracted content"""
    document: Document
    extracted_text: str
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    confidence: ConfidenceScore
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    
    # Extracted business information
    business_information: Optional[Dict[str, Any]] = None
    key_metrics: Optional[Dict[str, float]] = None
    financial_data: Optional[Dict[str, Any]] = None

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
    last_updated: datetime = Field(default_factory=datetime.now)

class CompetitiveAnalysis(BaseModel):
    """Competitive landscape analysis"""
    competitors: List[CompetitorProfile] = Field(default_factory=list)
    market_position: str = Field(..., description="Startup's position in market")
    competitive_advantages: List[str] = Field(default_factory=list)
    competitive_threats: List[str] = Field(default_factory=list)
    
    # Scoring
    competitive_strength_score: float = Field(..., ge=0, le=1)
    market_differentiation_score: float = Field(..., ge=0, le=1)
    
    # Analysis details
    total_competitors_analyzed: int = Field(..., ge=0)
    analysis_methodology: str
    confidence: ConfidenceScore

class MarketAnalysis(BaseModel):
    """Complete market analysis results"""
    market_size: MarketSizeData
    growth_rate: float = Field(..., description="Annual market growth rate")
    market_trends: List[str] = Field(default_factory=list)
    market_opportunities: List[str] = Field(default_factory=list)
    market_threats: List[str] = Field(default_factory=list)
    
    # Scoring
    market_opportunity_score: float = Field(..., ge=0, le=1)
    market_timing_score: float = Field(..., ge=0, le=1)
    
    # Analysis metadata
    analysis_date: datetime = Field(default_factory=datetime.now)
    data_sources: List[str] = Field(default_factory=list)
    confidence: ConfidenceScore

class RiskFactor(BaseModel):
    """Individual risk factor"""
    category: str = Field(..., description="Risk category")
    description: str = Field(..., min_length=10, max_length=500)
    severity: str = Field(..., description="Risk severity: low, medium, high, critical")
    probability: float = Field(..., ge=0, le=1, description="Probability of occurrence")
    impact: str = Field(..., description="Potential impact description")
    mitigation: Optional[str] = Field(None, description="Suggested mitigation strategy")
    
    @validator('severity')
    def validate_severity(cls, v):
        if v not in ['low', 'medium', 'high', 'critical']:
            raise ValueError('Severity must be low, medium, high, or critical')
        return v

class Recommendation(BaseModel):
    """Investment recommendation"""
    type: str = Field(..., description="Recommendation type")
    description: str = Field(..., min_length=10, max_length=1000)
    priority: str = Field(..., description="Priority: low, medium, high")
    rationale: str = Field(..., min_length=10, max_length=1000)
    expected_impact: Optional[str] = None
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ['low', 'medium', 'high']:
            raise ValueError('Priority must be low, medium, or high')
        return v

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