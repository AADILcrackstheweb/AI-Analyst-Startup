# src/workflow/langgraph_workflow.py

import asyncio
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
from langgraph import StateGraph, END
from langgraph.graph import Graph
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

from src.tools.web_search_tool import WebSearchTool
from src.tools.document_retriever_tool import DocumentRetrieverTool
from src.tools.sentiment_analysis_tool import SentimentAnalysisTool
from src.tools.competitor_analysis_tool import CompetitorAnalysisTool
from src.tools.market_size_estimator import MarketSizeEstimatorTool
from src.data.models import (
    StartupProfile, DueDiligenceResults, MarketAnalysis, 
    SentimentData, CompetitiveAnalysis, ConfidenceScore
)
from config.settings import get_settings

class WorkflowState(TypedDict):
    """State management for the due diligence workflow"""
    startup_profile: Optional[Dict[str, Any]]
    documents: List[Dict[str, Any]]
    web_search_results: Optional[Dict[str, Any]]
    document_analysis: Optional[Dict[str, Any]]
    sentiment_analysis: Optional[Dict[str, Any]]
    competitor_analysis: Optional[Dict[str, Any]]
    market_analysis: Optional[Dict[str, Any]]
    final_results: Optional[Dict[str, Any]]
    errors: List[str]
    execution_metadata: Dict[str, Any]
    messages: List[BaseMessage]

class DueDiligenceWorkflow:
    """LangGraph workflow for due diligence analysis"""
    
    def __init__(self):
        self.settings = get_settings()
        self.tools = self._initialize_tools()
        self.workflow = self._build_workflow()
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize all analysis tools"""
        return {
            "web_search": WebSearchTool(),
            "document_retriever": DocumentRetrieverTool(),
            "sentiment_analysis": SentimentAnalysisTool(),
            "competitor_analysis": CompetitorAnalysisTool(),
            "market_size_estimator": MarketSizeEstimatorTool()
        }
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_analysis)
        workflow.add_node("web_search", self._perform_web_search)
        workflow.add_node("document_processing", self._process_documents)
        workflow.add_node("sentiment_analysis", self._analyze_sentiment)
        workflow.add_node("competitor_analysis", self._analyze_competitors)
        workflow.add_node("market_analysis", self._analyze_market)
        workflow.add_node("synthesize_results", self._synthesize_results)
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("finalize", self._finalize_results)
        
        # Define the workflow edges
        workflow.set_entry_point("initialize")
        
        # Sequential flow with conditional branching
        workflow.add_edge("initialize", "web_search")
        workflow.add_edge("initialize", "document_processing")
        
        # Parallel analysis phase
        workflow.add_edge("web_search", "sentiment_analysis")
        workflow.add_edge("web_search", "competitor_analysis")
        workflow.add_edge("web_search", "market_analysis")
        
        workflow.add_edge("document_processing", "sentiment_analysis")
        workflow.add_edge("document_processing", "competitor_analysis")
        workflow.add_edge("document_processing", "market_analysis")
        
        # Synthesis phase
        workflow.add_edge("sentiment_analysis", "synthesize_results")
        workflow.add_edge("competitor_analysis", "synthesize_results")
        workflow.add_edge("market_analysis", "synthesize_results")
        
        # Quality check and finalization
        workflow.add_edge("synthesize_results", "quality_check")
        workflow.add_conditional_edges(
            "quality_check",
            self._should_retry,
            {
                "retry": "web_search",
                "continue": "finalize"
            }
        )
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def _initialize_analysis(self, state: WorkflowState) -> WorkflowState:
        """Initialize the analysis workflow"""
        
        state["execution_metadata"] = {
            "start_time": datetime.now().isoformat(),
            "workflow_version": "1.0",
            "tools_initialized": True
        }
        
        state["messages"].append(
            AIMessage(content="Initializing due diligence analysis workflow...")
        )
        
        # Validate startup profile
        if not state.get("startup_profile"):
            state["errors"].append("No startup profile provided")
            return state
        
        # Log initialization
        print(f"Initialized analysis for: {state['startup_profile'].get('company_name', 'Unknown')}")
        
        return state
    
    async def _perform_web_search(self, state: WorkflowState) -> WorkflowState:
        """Perform web search for external information"""
        
        try:
            startup_profile = state["startup_profile"]
            company_name = startup_profile.get("company_name", "")
            industry = startup_profile.get("industry", "")
            
            # Prepare search queries
            search_queries = [
                f"{company_name} company information",
                f"{company_name} funding news",
                f"{company_name} market analysis",
                f"{industry} market trends",
                f"{company_name} competitors"
            ]
            
            # Execute web search
            web_search_tool = self.tools["web_search"]
            search_input = {
                "queries": search_queries,
                "max_results_per_query": 10,
                "include_news": True,
                "include_social_media": True
            }
            
            results = await web_search_tool.execute(search_input)
            state["web_search_results"] = results
            
            state["messages"].append(
                AIMessage(content=f"Completed web search with {len(search_queries)} queries")
            )
            
        except Exception as e:
            error_msg = f"Web search failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
        
        return state
    
    async def _process_documents(self, state: WorkflowState) -> WorkflowState:
        """Process startup documents"""
        
        try:
            documents = state.get("documents", [])
            
            if not documents:
                state["messages"].append(
                    AIMessage(content="No documents provided for processing")
                )
                return state
            
            # Execute document processing
            doc_tool = self.tools["document_retriever"]
            doc_input = {
                "documents": documents,
                "extract_business_info": True,
                "ocr_enabled": True
            }
            
            results = await doc_tool.execute(doc_input)
            state["document_analysis"] = results
            
            state["messages"].append(
                AIMessage(content=f"Processed {len(documents)} documents")
            )
            
        except Exception as e:
            error_msg = f"Document processing failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
        
        return state
    
    async def _analyze_sentiment(self, state: WorkflowState) -> WorkflowState:
        """Analyze sentiment from various sources"""
        
        try:
            # Collect texts for sentiment analysis
            texts = []
            sources = []
            
            # Add web search content
            if state.get("web_search_results"):
                web_results = state["web_search_results"].get("search_results", [])
                for result in web_results:
                    if result.get("snippet"):
                        texts.append(result["snippet"])
                        sources.append(f"web_{result.get('source', 'unknown')}")
            
            # Add document content
            if state.get("document_analysis"):
                doc_results = state["document_analysis"].get("processed_documents", [])
                for doc in doc_results:
                    if doc.get("extracted_text"):
                        texts.append(doc["extracted_text"][:1000])  # Limit text length
                        sources.append(f"document_{doc.get('name', 'unknown')}")
            
            if not texts:
                state["messages"].append(
                    AIMessage(content="No text content available for sentiment analysis")
                )
                return state
            
            # Execute sentiment analysis
            sentiment_tool = self.tools["sentiment_analysis"]
            sentiment_input = {
                "texts": texts,
                "sources": sources,
                "include_social_media": True,
                "include_news": True,
                "company_name": state["startup_profile"].get("company_name")
            }
            
            results = await sentiment_tool.execute(sentiment_input)
            state["sentiment_analysis"] = results
            
            state["messages"].append(
                AIMessage(content=f"Analyzed sentiment for {len(texts)} text sources")
            )
            
        except Exception as e:
            error_msg = f"Sentiment analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
        
        return state
    
    async def _analyze_competitors(self, state: WorkflowState) -> WorkflowState:
        """Analyze competitive landscape"""
        
        try:
            startup_profile = state["startup_profile"]
            
            # Execute competitor analysis
            competitor_tool = self.tools["competitor_analysis"]
            competitor_input = {
                "company_name": startup_profile.get("company_name", ""),
                "industry": startup_profile.get("industry"),
                "target_market": startup_profile.get("target_market"),
                "product_category": startup_profile.get("product_category"),
                "include_funding_data": True,
                "include_social_metrics": True,
                "max_competitors": 15
            }
            
            results = await competitor_tool.execute(competitor_input)
            state["competitor_analysis"] = results
            
            competitor_count = len(results.get("competitors", []))
            state["messages"].append(
                AIMessage(content=f"Analyzed {competitor_count} competitors")
            )
            
        except Exception as e:
            error_msg = f"Competitor analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
        
        return state
    
    async def _analyze_market(self, state: WorkflowState) -> WorkflowState:
        """Analyze market size and opportunities"""
        
        try:
            startup_profile = state["startup_profile"]
            
            # Execute market analysis
            market_tool = self.tools["market_size_estimator"]
            market_input = {
                "industry": startup_profile.get("industry", ""),
                "product_category": startup_profile.get("product_category"),
                "target_geography": startup_profile.get("target_geography", "Global"),
                "time_horizon": 5,
                "company_stage": startup_profile.get("stage", "startup"),
                "include_tam_sam_som": True,
                "include_growth_projections": True,
                "include_market_trends": True
            }
            
            results = await market_tool.execute(market_input)
            state["market_analysis"] = results
            
            market_size = results.get("current_market_size_usd", 0)
            state["messages"].append(
                AIMessage(content=f"Analyzed market size: ${market_size:,.0f}")
            )
            
        except Exception as e:
            error_msg = f"Market analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
        
        return state
    
    async def _synthesize_results(self, state: WorkflowState) -> WorkflowState:
        """Synthesize all analysis results"""
        
        try:
            # Collect all analysis results
            analyses = {
                "web_search": state.get("web_search_results"),
                "documents": state.get("document_analysis"),
                "sentiment": state.get("sentiment_analysis"),
                "competitors": state.get("competitor_analysis"),
                "market": state.get("market_analysis")
            }
            
            # Calculate overall confidence
            confidences = []
            for analysis in analyses.values():
                if analysis and analysis.get("confidence"):
                    confidences.append(analysis["confidence"]["score"])
            
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(state, analyses)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(state, analyses)
            
            # Create synthesized results
            synthesized = {
                "executive_summary": executive_summary,
                "recommendations": recommendations,
                "overall_confidence": {
                    "score": overall_confidence,
                    "level": "high" if overall_confidence >= 0.7 else "medium" if overall_confidence >= 0.4 else "low"
                },
                "analysis_completeness": self._assess_completeness(analyses),
                "synthesis_timestamp": datetime.now().isoformat()
            }
            
            state["final_results"] = synthesized
            
            state["messages"].append(
                AIMessage(content="Successfully synthesized all analysis results")
            )
            
        except Exception as e:
            error_msg = f"Result synthesis failed: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
        
        return state
    
    def _generate_executive_summary(self, state: WorkflowState, analyses: Dict[str, Any]) -> str:
        """Generate executive summary from all analyses"""
        
        company_name = state["startup_profile"].get("company_name", "Unknown Company")
        industry = state["startup_profile"].get("industry", "Unknown Industry")
        
        summary_parts = [
            f"Due Diligence Analysis for {company_name}",
            f"Industry: {industry}",
            ""
        ]
        
        # Market analysis summary
        if analyses.get("market"):
            market = analyses["market"]
            market_size = market.get("current_market_size_usd", 0)
            cagr = market.get("cagr_percentage", 0)
            summary_parts.append(
                f"Market Size: ${market_size:,.0f} with {cagr:.1f}% CAGR"
            )
        
        # Competitive landscape summary
        if analyses.get("competitors"):
            competitors = analyses["competitors"]
            competitor_count = len(competitors.get("competitors", []))
            summary_parts.append(
                f"Competitive Landscape: {competitor_count} key competitors identified"
            )
        
        # Sentiment summary
        if analyses.get("sentiment"):
            sentiment = analyses["sentiment"]
            overall_sentiment = sentiment.get("overall_sentiment", {})
            sentiment_label = overall_sentiment.get("label", "neutral")
            summary_parts.append(
                f"Market Sentiment: {sentiment_label.title()}"
            )
        
        return "\n".join(summary_parts)
    
    def _generate_recommendations(self, state: WorkflowState, analyses