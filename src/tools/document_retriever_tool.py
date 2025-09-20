# src/tools/document_retriever_tool.py (Complete version)

import io
import time
import PyPDF2
import docx
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from google.cloud import storage, documentai_v1 as documentai
from google.cloud import vision

from .base_tool import BaseDueDiligenceTool
from src.data.models import Document, ProcessedDocument, ConfidenceScore, DocumentType

class DocumentRetrievalInput(BaseModel):
    """Input model for document retrieval"""
    documents: List[Document] = Field(..., min_items=1, description="Documents to process")
    extract_business_info: bool = Field(default=True, description="Extract business information")
    ocr_enabled: bool = Field(default=True, description="Enable OCR for scanned documents")
    
    @validator('documents')
    def validate_documents(cls, v):
        if not v:
            raise ValueError('At least one document must be provided')
        return v

class ExtractedBusinessInfo(BaseModel):
    """Extracted business information from documents"""
    company_name: Optional[str] = None
    industry: Optional[str] = None
    product_description: Optional[str] = None
    target_market: Optional[str] = None
    business_model: Optional[str] = None
    key_metrics: Dict[str, float] = Field(default_factory=dict)
    financial_data: Dict[str, Any] = Field(default_factory=dict)
    team_information: List[str] = Field(default_factory=list)
    funding_information: Dict[str, Any] = Field(default_factory=dict)

class DocumentRetrievalOutput(BaseModel):
    """Output model for document retrieval"""
    processed_documents: List[ProcessedDocument] = Field(default_factory=list)
    business_information: Optional[ExtractedBusinessInfo] = None
    total_documents: int = Field(..., ge=0)
    successfully_processed: int = Field(..., ge=0)
    processing_errors: List[str] = Field(default_factory=list)
    confidence: ConfidenceScore = Field(...)
    execution_time: float = Field(..., ge=0)

class DocumentRetrieverTool(BaseDueDiligenceTool):
    """Document retrieval tool with Pydantic validation"""
    
    name = "document_retriever_tool"
    description = "Process and extract information from startup documents"
    input_model = DocumentRetrievalInput
    output_model = DocumentRetrievalOutput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage_client = storage.Client()
        try:
            self.document_ai_client = documentai.DocumentProcessorServiceClient()
        except Exception:
            self.document_ai_client = None
            self.logger.warning("Document AI client not available")
        
        try:
            self.vision_client = vision.ImageAnnotatorClient()
        except Exception:
            self.vision_client = None
            self.logger.warning("Vision API client not available")
    
    async def _execute(self, validated_input: DocumentRetrievalInput) -> Dict[str, Any]:
        """Execute document processing with validated input"""
        start_time = time.time()
        
        processed_documents = []
        processing_errors = []
        
        for document in validated_input.documents:
            try:
                # Process individual document
                processed_doc = await self._process_single_document(
                    document, 
                    validated_input.ocr_enabled
                )
                processed_documents.append(processed_doc)
                
            except Exception as e:
                error_msg = f"Failed to process {document.name}: {str(e)}"
                processing_errors.append(error_msg)
                self.logger.error(error_msg)
        
        # Extract business information if requested
        business_info = None
        if validated_input.extract_business_info and processed_documents:
            business_info = await self._extract_business_information(processed_documents)
        
        # Calculate confidence
        success_rate = len(processed_documents) / len(validated_input.documents)
        confidence_score = success_rate * 0.8 + (0.2 if business_info else 0)
        confidence = ConfidenceScore(
            score=confidence_score,
            level="high" if confidence_score >= 0.7 else "medium" if confidence_score >= 0.4 else "low"
        )
        
        execution_time = time.time() - start_time
        
        return {
            "processed_documents": [doc.dict() for doc in processed_documents],
            "business_information": business_info.dict() if business_info else None,
            "total_documents": len(validated_input.documents),
            "successfully_processed": len(processed_documents),
            "processing_errors": processing_errors,
            "confidence": confidence.dict(),
            "execution_time": execution_time
        }
    
    def _extract_financial_data(self, text: str) -> Dict[str, Any]:
        """Extract financial data from text"""
        financial_data = {}
        
        # Revenue patterns
        revenue_patterns = [
            r'revenue[:\s]+\$?([\d,]+(?:\.\d+)?)[mk]?',
            r'sales[:\s]+\$?([\d,]+(?:\.\d+)?)[mk]?',
            r'income[:\s]+\$?([\d,]+(?:\.\d+)?)[mk]?'
        ]
        
        for pattern in revenue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                financial_data['revenue'] = matches[0]
                break
        
        # Funding patterns
        funding_patterns = [
            r'raised[:\s]+\$?([\d,]+(?:\.\d+)?)[mk]?',
            r'funding[:\s]+\$?([\d,]+(?:\.\d+)?)[mk]?',
            r'investment[:\s]+\$?([\d,]+(?:\.\d+)?)[mk]?'
        ]
        
        for pattern in funding_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                financial_data['funding'] = matches[0]
                break
        
        return financial_data
    
    async def _extract_business_information(self, processed_docs: List[ProcessedDocument]) -> ExtractedBusinessInfo:
        """Extract and consolidate business information from processed documents"""
        
        # Combine all extracted text
        combined_text = " ".join([doc.extracted_text for doc in processed_docs])
        
        # Extract information using simple patterns (in production, use LLM)
        business_info = {}
        
        # Company name extraction
        company_patterns = [
            r'company[:\s]+([A-Z][a-zA-Z\s]+)',
            r'startup[:\s]+([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:Inc|LLC|Corp|Ltd)'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, combined_text)
            if matches:
                business_info['company_name'] = matches[0].strip()
                break
        
        # Industry extraction
        industry_keywords = {
            'saas': ['software', 'saas', 'platform', 'cloud'],
            'fintech': ['financial', 'fintech', 'banking', 'payment'],
            'healthtech': ['health', 'medical', 'healthcare', 'biotech'],
            'ecommerce': ['ecommerce', 'retail', 'marketplace', 'shopping']
        }
        
        text_lower = combined_text.lower()
        for industry, keywords in industry_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                business_info['industry'] = industry
                break
        
        # Extract team information
        team_patterns = [
            r'(?:founder|ceo|cto|cfo)[:\s]+([A-Z][a-zA-Z\s]+)',
            r'team[:\s]+([A-Z][a-zA-Z\s,]+)'
        ]
        
        team_members = []
        for pattern in team_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            team_members.extend(matches)
        
        business_info['team_information'] = list(set(team_members))
        
        # Combine financial data from all documents
        all_financial_data = {}
        for doc in processed_docs:
            if doc.financial_data:
                all_financial_data.update(doc.financial_data)
        
        business_info['financial_data'] = all_financial_data
        
        return ExtractedBusinessInfo(**business_info)