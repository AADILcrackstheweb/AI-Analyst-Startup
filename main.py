from fastapi import FastAPI
from pydantic import BaseModel
from src.agents.due_diligence_agent import due_diligence_agent_executor
import uvicorn

app = FastAPI()

class StartupIdea(BaseModel):
    idea: str

@app.post("/analyze")
async def analyze_startup(startup_idea: StartupIdea):
    result = due_diligence_agent_executor.invoke({
        "startup_idea": startup_idea.idea
    })
    return {"analysis": result["output"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
