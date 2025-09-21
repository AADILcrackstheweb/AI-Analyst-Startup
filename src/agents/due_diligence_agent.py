from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from src.tools.due_diligence_tools import tools

# Load the prompt from the file
with open("src/prompts/due_diligence_agent_prompt.txt", "r") as f:
    prompt_template_str = f.read()

# Initialize the language model
default_params = {
    "max_output_tokens": 1024,
    "temperature": 0,
    "top_p": 0.2,
    "top_k": 1,
    "model_name": "gemini-2.0-flash",
    "project": "aianalyst-472718"
}
llm = VertexAI(**default_params)

# Create the prompt template
prompt = PromptTemplate.from_template(prompt_template_str)

# Create the ReAct agent
due_diligence_agent = create_react_agent(llm, tools, prompt)
# due_diligence_agent = ReActAgent.from_llm_and_tools(
#     llm=llm,
#     tools=tools,
#     prompt=prompt_template
# )

# Create the agent executor
due_diligence_agent_executor = AgentExecutor(
    agent=due_diligence_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
# AgentExecutor.from_agent_and_tools(
#     agent=due_diligence_agent,
#     tools=tools,
#     verbose=True,
#     handle_parsing_errors=True
# )
