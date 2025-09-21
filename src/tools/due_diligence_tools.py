from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import Tool
from data.models import CustomerSegments, MarketAnalysis, CompetitiveAnalysis

default_params = {
    "max_output_tokens": 1024,
    "temperature": 0,
    "top_p": 0.2,
    "top_k": 1,
    "model_name": "gemini-2.0-flash"
}
llm = VertexAI(**default_params)

# Customer Segments Chain
customer_parser = PydanticOutputParser(pydantic_object=CustomerSegments)

customer_prompt = PromptTemplate(
    template="""
    You are a market analyst. Analyze the customer segments for this startup idea:

    {startup_idea}

    {format_instructions}
    """,
    input_variables=["startup_idea"],
    partial_variables={"format_instructions": customer_parser.get_format_instructions()}
)

customer_chain = customer_prompt | llm | customer_parser 

def customer_segments_tool(startup_idea: str):
    try:
        result = customer_chain.invoke(startup_idea=startup_idea)
        return result 
    except:
        raise ValueError("Error analyzing customer segments")

# Market Size Chain
market_size_parser = PydanticOutputParser(pydantic_object=MarketAnalysis)

market_size_prompt = PromptTemplate(
    template="""
    Estimate TAM, SAM, and SOM for this startup idea:

    {startup_idea}

    {format_instructions}
    """,
    input_variables=["startup_idea"],
    partial_variables={"format_instructions": market_size_parser.get_format_instructions()}
)

market_size_chain = market_size_prompt | llm | market_size_parser

def market_size_tool(startup_idea: str):
    try:
        result = market_size_chain.invoke(startup_idea=startup_idea)
        return result 
    except:
        raise ValueError("Error analyzing market size")

# Competitive Landscape Chain
landscape_parser = PydanticOutputParser(pydantic_object=CompetitiveAnalysis)

landscape_prompt = PromptTemplate(
    template="""
    Identify competitors and alternatives for this startup idea:

    {startup_idea}

    {format_instructions}
    """,
    input_variables=["startup_idea"],
    partial_variables={"format_instructions": landscape_parser.get_format_instructions()}
)

landscape_chain = landscape_prompt | llm | landscape_parser

def competitive_landscape_tool(startup_idea: str):
    try:
        result = landscape_chain.invoke(startup_idea=startup_idea)
        return result
    except:
        raise ValueError("Error analyzing competitive landscape")


tools = [
    Tool(name="Customer Segments", func=customer_segments_tool, description="Analyzes a startup idea and identifies potential customer segments, including primary, secondary and detailed characteristics."),
    Tool(name="Market Size", func=market_size_tool, description="Estimates TAM/SAM/SOM"),
    Tool(name="Competitive Landscape", func=competitive_landscape_tool, description="Maps competitors")
]


