from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent

# Schemas
from data.models import CustomerSegments, MarketSize, CompetitiveLandscape, MarketAnalysis

llm = ChatOpenAI(model="gpt-4o", temperature=0)

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

customer_chain = customer_prompt | llm | customer_parser #LLMChain(llm=llm, prompt=customer_prompt)

def customer_segments_tool(startup_idea: str) -> dict:
    result = customer_chain.invoke(startup_idea=startup_idea)
    return result #customer_parser.parse(result).dict()

# 2️⃣ Market Size Chain
market_size_parser = PydanticOutputParser(pydantic_object=MarketSize)

market_size_prompt = PromptTemplate(
    template="""
    Estimate TAM, SAM, and SOM for this startup idea:

    {startup_idea}

    {format_instructions}
    """,
    input_variables=["startup_idea"],
    partial_variables={"format_instructions": market_size_parser.get_format_instructions()}
)

market_size_chain = LLMChain(llm=llm, prompt=market_size_prompt)

def market_size_tool(startup_idea: str) -> dict:
    result = market_size_chain.run(startup_idea=startup_idea)
    return market_size_parser.parse(result).dict()

# 3️⃣ Competitive Landscape Chain
landscape_parser = PydanticOutputParser(pydantic_object=CompetitiveLandscape)

landscape_prompt = PromptTemplate(
    template="""
    Identify competitors and alternatives for this startup idea:

    {startup_idea}

    {format_instructions}
    """,
    input_variables=["startup_idea"],
    partial_variables={"format_instructions": landscape_parser.get_format_instructions()}
)

landscape_chain = LLMChain(llm=llm, prompt=landscape_prompt)

def competitive_landscape_tool(startup_idea: str) -> dict:
    result = landscape_chain.run(startup_idea=startup_idea)
    return landscape_parser.parse(result).dict()

# 4️⃣ Wrap into Tools
tools = [
    Tool(name="Customer Segments", func=customer_segments_tool, description="Finds customer segments"),
    Tool(name="Market Size", func=market_size_tool, description="Estimates TAM/SAM/SOM"),
    Tool(name="Competitive Landscape", func=competitive_landscape_tool, description="Maps competitors")
]

# 5️⃣ Central Agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

startup_idea = "AI-powered fitness coaching app"
response = agent.run(f"Run full market analysis for: {startup_idea}")

print(response)
