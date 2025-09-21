from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
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
    
    @field_validator('company_name')
    def validate_company_name(cls, v):
        if not v.strip():
            raise ValueError('Company name cannot be empty')
        return v.strip()
    
    @field_validator('product_description', 'target_market')
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
    
    
    @field_validator('tam', 'sam', 'som')
    def validate_market_sizes(cls, v):
        if v < 0:
            raise ValueError('Market size cannot be negative')
        return v
    
    @model_validator(mode='after')
    def validate_market_hierarchy(cls, values):
        tam = values.get('tam', 0)
        sam = values.get('sam', 0)
        som = values.get('som', 0)
        
        if sam > tam:
            raise ValueError('SAM cannot be larger than TAM')
        if som > sam:
            raise ValueError('SOM cannot be larger than SAM')
        
        return values


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

class CustomerSegments(BaseModel):
    primary: List[str]
    secondary: List[str]
    details: str


class DueDiligenceResults(BaseModel):
    """Complete due diligence analysis results"""
    # Input data
    startup_profile: StartupProfile
    
    # Analysis results
    market_analysis: MarketAnalysis
    customer_analysis: CustomerSegments
    competitive_analysis: CompetitiveAnalysis


