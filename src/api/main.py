# src/api/main.py

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from contextlib import asynccontextmanager

from src.workflow.langgraph_workflow import DueDiligenceWorkflow, WorkflowState
from src.data.models import StartupProfile, Document, DueDiligenceResults
from config.settings import get_settings
from src.utils.logging_config import setup_logging
from src.utils.auth import verify_api_key
from src.utils.file_handler import FileHandler

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global workflow instance
workflow_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global workflow_instance
    
    # Startup
    logger.info("Starting Due Diligence Agent API")
    workflow_instance = DueDiligenceWorkflow()
    logger.info("Workflow initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Due Diligence Agent API")

# Create FastAPI app
app = FastAPI(
    title="Due Diligence Agent API",
    description="AI-powered due diligence analysis for startups",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Settings
settings = get_settings()

# File handler
file_handler = FileHandler()

# Request/Response Models
class AnalysisRequest(BaseModel):
    """Request model for due diligence analysis"""
    startup_profile: StartupProfile = Field(..., description="Startup profile information")
    analysis_options: Dict[str, bool] = Field(
        default_factory=lambda: {
            "include_web_search": True,
            "include_sentiment_analysis": True,
            "include_competitor_analysis": True,
            "include_market_analysis": True,
            "include_document_processing": True
        },
        description="Analysis options"
    )
    priority: str = Field(default="normal", description="Analysis priority (low/normal/high)")
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ["low", "normal", "high"]:
            raise ValueError('Priority must be one of: low, normal, high')
        return v

class AnalysisResponse(BaseModel):
    """Response model for analysis requests"""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: str = Field(..., description="Analysis status")
    message: str = Field(..., description="Status message")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")

class AnalysisStatus(BaseModel):
    """Analysis status model"""
    analysis_id: str = Field(..., description="Analysis identifier")
    status: str = Field(..., description="Current status")
    progress_percentage: float = Field(..., ge=0, le=100, description="Progress percentage")
    current_step: str = Field(..., description="Current processing step")
    steps_completed: List[str] = Field(default_factory=list, description="Completed steps")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    started_at: datetime = Field(..., description="Analysis start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

class AnalysisResults(BaseModel):
    """Analysis results model"""
    analysis_id: str = Field(..., description="Analysis identifier")
    startup_profile: StartupProfile = Field(..., description="Original startup profile")
    results: DueDiligenceResults = Field(..., description="Analysis results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Analysis metadata")
    completed_at: datetime = Field(..., description="Completion timestamp")

# In-memory storage for demo (use database in production)
analysis_storage: Dict[str, Dict[str, Any]] = {}
analysis_results: Dict[str, AnalysisResults] = {}

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Due Diligence Agent API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "workflow_initialized": workflow_instance is not None
    }

@app.post("/api/v1/analysis", response_model=AnalysisResponse)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Start a new due diligence analysis"""
    
    try:
        # Generate analysis ID
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(request.startup_profile.dict())) % 10000}"
        
        # Initialize analysis tracking
        analysis_storage[analysis_id] = {
            "status": "queued",
            "progress_percentage": 0.0,
            "current_step": "initialization",
            "steps_completed": [],
            "errors": [],
            "started_at": datetime.now(),
            "request": request.dict()
        }
        
        # Start background analysis
        background_tasks.add_task(
            run_analysis,
            analysis_id,
            request
        )
        
        logger.info(f"Started analysis {analysis_id} for {request.startup_profile.company_name}")
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="queued",
            message="Analysis started successfully",
            estimated_completion_time=datetime.now().replace(minute=datetime.now().minute + 15)  # 15 min estimate
        )
        
    except Exception as e:
        logger.error(f"Failed to start analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@app.get("/api/v1/analysis/{analysis_id}/status", response_model=AnalysisStatus)
async def get_analysis_status(analysis_id: str, api_key: str = Depends(verify_api_key)):
    """Get analysis status"""
    
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis_data = analysis_storage[analysis_id]
    
    return AnalysisStatus(
        analysis_id=analysis_id,
        status=analysis_data["status"],
        progress_percentage=analysis_data["progress_percentage"],
        current_step=analysis_data["current_step"],
        steps_completed=analysis_data["steps_completed"],
        errors=analysis_data["errors"],
        started_at=analysis_data["started_at"],
        estimated_completion=analysis_data.get("estimated_completion")
    )

@app.get("/api/v1/analysis/{analysis_id}/results", response_model=AnalysisResults)
async def get_analysis_results(analysis_id: str, api_key: str = Depends(verify_api_key)):
    """Get analysis results"""
    
    if analysis_id not in analysis_results:
        if analysis_id in analysis_storage:
            status = analysis_storage[analysis_id]["status"]
            if status in ["queued", "running"]:
                raise HTTPException(status_code=202, detail="Analysis still in progress")
            elif status == "failed":
                raise HTTPException(status_code=500, detail="Analysis failed")
        raise HTTPException(status_code=404, detail="Analysis results not found")
    
    return analysis_results[analysis_id]

@app.post("/api/v1/analysis/{analysis_id}/documents")
async def upload_documents(
    analysis_id: str,
    files: List[UploadFile] = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Upload documents for analysis"""
    
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        uploaded_documents = []
        
        for file in files:
            # Save file and create document record
            file_path = await file_handler.save_uploaded_file(file, analysis_id)
            
            document = Document(
                name=file.filename,
                file_path=file_path,
                file_type=file.content_type,
                file_size=file.size,
                upload_timestamp=datetime.now()
            )
            
            uploaded_documents.append(document.dict())
        
        # Update analysis storage with documents
        if "documents" not in analysis_storage[analysis_id]:
            analysis_storage[analysis_id]["documents"] = []
        
        analysis_storage[analysis_id]["documents"].extend(uploaded_documents)
        
        logger.info(f"Uploaded {len(files)} documents for analysis {analysis_id}")
        
        return {
            "message": f"Successfully uploaded {len(files)} documents",
            "documents": uploaded_documents
        }
        
    except Exception as e:
        logger.error(f"Failed to upload documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload documents: {str(e)}")

@app.delete("/api/v1/analysis/{analysis_id}")
async def cancel_analysis(analysis_id: str, api_key: str = Depends(verify_api_key)):
    """Cancel an ongoing analysis"""
    
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis_data = analysis_storage[analysis_id]
    
    if analysis_data["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel analysis with status: {analysis_data['status']}")
    
    # Update status
    analysis_storage[analysis_id]["status"] = "cancelled"
    analysis_storage[analysis_id]["current_step"] = "cancelled"
    
    logger.info(f"Cancelled analysis {analysis_id}")
    
    return {"message": "Analysis cancelled successfully"}

@app.get("/api/v1/analyses")
async def list_analyses(
    status: Optional[str] = None,
    limit: int = 50,
    api_key: str = Depends(verify_api_key)
):
    """List all analyses"""
    
    analyses = []
    
    for analysis_id, data in analysis_storage.items():
        if status is None or data["status"] == status:
            analyses.append({
                "analysis_id": analysis_id,
                "status": data["status"],
                "company_name": data["request"]["startup_profile"]["company_name"],
                "started_at": data["started_at"],
                "progress_percentage": data["progress_percentage"]
            })
    
    # Sort by start time (newest first)
    analyses.sort(key=lambda x: x["started_at"], reverse=True)
    
    return {
        "analyses": analyses[:limit],
        "total_count": len(analyses)
    }

# Background task for running analysis
async def run_analysis(analysis_id: str, request: AnalysisRequest):
    """Run the due diligence analysis in the background"""
    
    try:
        # Update status
        analysis_storage[analysis_id]["status"] = "running"
        analysis_storage[analysis_id]["current_step"] = "initializing"
        analysis_storage[analysis_id]["progress_percentage"] = 5.0
        
        # Prepare workflow state
        initial_state: WorkflowState = {
            "startup_profile": request.startup_profile.dict(),
            "documents": analysis_storage[analysis_id].get("documents", []),
            "web_search_results": None,
            "document_analysis": None,
            "sentiment_analysis": None,
            "competitor_analysis": None,
            "market_analysis": None,
            "final_results": None,
            "errors": [],
            "execution_metadata": {},
            "messages": []
        }
        
        # Run workflow
        logger.info(f"Starting workflow execution for analysis {analysis_id}")
        
        # Execute workflow steps with progress updates
        final_state = await execute_workflow_with_progress(
            workflow_instance.workflow,
            initial_state,
            analysis_id
        )
        
        # Create results
        if final_state.get("final_results") and not final_state.get("errors"):
            # Convert to DueDiligenceResults
            results = create_due_diligence_results(final_state, request.startup_profile)
            
            analysis_results[analysis_id] = AnalysisResults(
                analysis_id=analysis_id,
                startup_profile=request.startup_profile,
                results=results,
                metadata=final_state.get("execution_metadata", {}),
                completed_at=datetime.now()
            )
            
            # Update status
            analysis_storage[analysis_id]["status"] = "completed"
            analysis_storage[analysis_id]["progress_percentage"] = 100.0
            analysis_storage[analysis_id]["current_step"] = "completed"
            
            logger.info(f"Successfully completed analysis {analysis_id}")
            
        else:
            # Analysis failed
            analysis_storage[analysis_id]["status"] = "failed"
            analysis_storage[analysis_id]["errors"].extend(final_state.get("errors", []))
            
            logger.error(f"Analysis {analysis_id} failed with errors: {final_state.get('errors', [])}")
    
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error in analysis {analysis_id}: {str(e)}"
        logger.error(error_msg)
        
        analysis_storage[analysis_id]["status"] = "failed"
        analysis_storage[analysis_id]["errors"].append(error_msg)

async def execute_workflow_with_progress(workflow, initial_state: WorkflowState, analysis_id: str) -> WorkflowState:
    """Execute workflow with progress tracking"""
    
    # Define progress milestones
    progress_steps = {
        "initialize": 10,
        "web_search": 25,
        "document_processing": 40,
        "sentiment_analysis": 55,
        "competitor_analysis": 70,
        "market_analysis": 85,
        "synthesize_results": 95,
        "finalize": 100
    }
    
    # Execute workflow (simplified - in production, implement proper step tracking)
    try:
        final_state = await workflow.ainvoke(initial_state)
        
        # Update progress based on completion
        analysis_storage[analysis_id]["progress_percentage"] = 100.0
        analysis_storage[analysis_id]["current_step"] = "completed"
        analysis_storage[analysis_id]["steps_completed"] = list(progress_steps.keys())
        
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        raise

def create_due_diligence_results(final_state: WorkflowState, startup_profile: StartupProfile) -> DueDiligenceResults:
    """Create DueDiligenceResults from workflow state"""
    
    # Extract results from final state
    final_results