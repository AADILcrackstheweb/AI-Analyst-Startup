from fastapi import FastAPI
from pydantic import BaseModel
from src.agents.due_diligence_agent import due_diligence_agent_executor
import uvicorn
from google.cloud import storage
import json

app = FastAPI()

class StartupIdea(BaseModel):
    url: str

@app.post("/analyze")
async def analyze_startup(startup_idea: StartupIdea):
    gcs_path = startup_idea.url
    
    # Assuming GCS path is in the format "gs://bucket_name/file_name.json"
    if gcs_path.startswith("gs://"):
        path_parts = gcs_path.replace("gs://", "").split("/", 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        try:
            # Download the file contents as a string
            json_content_str = blob.download_as_string()
            # Parse the JSON string
            # json_content = json.loads(json_content_str)
            
            # Extract the content to be used as startup_idea
            # Assuming the JSON has a key "idea" that contains the startup description
            idea_content = json_content_str #json_content.get("idea", "")
        except Exception as e:
            return {"error": f"Failed to read or parse GCS file: {e}"}
    else:
        # If it's not a GCS path, use the content directly
        idea_content = gcs_path

    result = due_diligence_agent_executor.invoke({
        "startup_idea": idea_content
    })
    return {"analysis": result["output"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
