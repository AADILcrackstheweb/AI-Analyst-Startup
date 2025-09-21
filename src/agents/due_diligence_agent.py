from typing import Dict, List, Any, Annotated
from uuid import UUID
from langchain_google_vertexai import VertexAI
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.checkpoint import BaseCheckpoint

from src.tools.due_diligence_tools import tools
from src.data.models import (
    CustomerSegments, 
    MarketAnalysis, 
    CompetitiveAnalysis
)

# Load due diligence prompt template
with open("src/prompts/due_diligence_agent_prompt.txt", "r") as f:
    DUE_DILIGENCE_PROMPT = f.read()

# Initialize LLM
llm = VertexAI(
    max_output_tokens=1024,
    temperature=0.1,
    top_p=0.8,
    model_name="gemini-2.0-pro"
)

class DueDiligenceState(dict):
    """State object for the due diligence workflow"""
    startup_idea: str
    messages: List[BaseMessage]
    tools_output: Dict[str, Any]
    final_analysis: Dict[str, Any]

# Tool executor node
tool_executor = ToolExecutor(tools)

def should_continue(state: DueDiligenceState) -> bool:
    """Check if we should continue running tools based on last message"""
    last_message = state["messages"][-1].content
    
    # Check if we have all required analysis components
    required_sections = [
        "Customer Segments",
        "Market Size",
        "Market Trends",
        "Opportunities & Gaps",
        "Competitive Landscape",
        "Customer Pain Points",
        "Monetization & Business Models",
        "Risks & Challenges"
    ]
    
    return not all(section in str(state.get("final_analysis", {})) for section in required_sections)

def agent_node(state: DueDiligenceState) -> DueDiligenceState:
    """Main agent node that decides what to do next"""
    
    # Format prompt with current state
    prompt = DUE_DILIGENCE_PROMPT.format(
        startup_idea=state["startup_idea"],
        format_instructions="Use the available tools to conduct the analysis."
    )
    
    # Get agent's next action
    messages = [HumanMessage(content=prompt)]
    if state.get("messages"):
        messages.extend(state["messages"])
    
    response = llm.invoke(messages)
    state["messages"] = messages + [response]
    
    return state

def tools_node(state: DueDiligenceState) -> Dict[str, Any]:
    """Execute any tools requested in the last message"""
    last_message = state["messages"][-1].content
    tool_calls = []
    
    # Extract tool calls based on the agent's last message
    # This is simplified - in practice you'd want more robust tool call extraction
    for tool in tools:
        if tool.name.lower() in last_message.lower():
            tool_calls.append({
                "tool": tool.name,
                "input": {"startup_idea": state["startup_idea"]}
            })
    
    if tool_calls:
        results = tool_executor.execute(tool_calls)
        return {"tool_output": results}
    return {}

def compile_analysis_node(state: DueDiligenceState) -> DueDiligenceState:
    """Compile final analysis from tool outputs and agent messages"""
    tools_output = state.get("tools_output", {})
    
    analysis = {
        "Customer Segments": tools_output.get("Customer Segments", {}),
        "Market Size": tools_output.get("Market Size", {}),
        "Competitive Landscape": tools_output.get("Competitive Landscape", {}),
    }
    
    # Extract other sections from agent messages
    for message in state["messages"]:
        content = str(message.content)
        if "Market Trends:" in content:
            analysis["Market Trends"] = content.split("Market Trends:")[1].split("\n")[0].strip()
        if "Opportunities & Gaps:" in content:
            analysis["Opportunities & Gaps"] = content.split("Opportunities & Gaps:")[1].split("\n")[0].strip()
        # Add other sections similarly
    
    state["final_analysis"] = analysis
    return state

def create_due_diligence_graph() -> StateGraph:
    """Create the LangGraph workflow for due diligence analysis"""
    
    workflow = StateGraph(DueDiligenceState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("compile", compile_analysis_node)
    
    # Add edges
    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "compile")
    workflow.add_conditional_edges(
        "compile",
        should_continue,
        {
            True: "agent",
            False: END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    return workflow

class DueDiligenceAgent:
    """Due Diligence Agent using LangGraph for workflow orchestration"""
    
    def __init__(self):
        self.graph = create_due_diligence_graph()
    
    async def analyze(self, startup_idea: str) -> Dict[str, Any]:
        """Run due diligence analysis on a startup idea"""
        
        initial_state = DueDiligenceState({
            "startup_idea": startup_idea,
            "messages": [],
            "tools_output": {},
            "final_analysis": {}
        })
        
        for output in self.graph.stream(initial_state):
            state = output.state
            
            
        return state["final_analysis"]