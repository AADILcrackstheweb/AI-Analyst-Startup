from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.agents import ReActAgent, AgentExecutor
from src.tools.due_diligence_tools import tools

# Load the prompt from the file
with open("src/prompts/due_diligence_agent_prompt.txt", "r") as f:
    prompt_template_str = f.read()

# Initialize the language model
llm = VertexAI(model_name="gemini-2.0-flash", temperature=0)

# Create the prompt template
prompt_template = PromptTemplate.from_template(prompt_template_str)

# Create the ReAct agent
due_diligence_agent = ReActAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

# Create the agent executor
due_diligence_agent_executor = AgentExecutor.from_agent_and_tools(
    agent=due_diligence_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
