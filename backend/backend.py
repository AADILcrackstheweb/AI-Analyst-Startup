from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import hashlib, json, os

app = FastAPI()

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Hardcoded values
BUCKET_NAME = "your bucket name"
PROJECT_ID = "your pid"
storage_client = storage.Client(project=PROJECT_ID)

def generate_uid(company_name: str) -> str:
    return hashlib.md5(company_name.encode()).hexdigest()[:12]

@app.get("/")
def read_root():
    return {"message": "Backend is working!"}

@app.post("/upload")
async def upload_files(company_name: str = Form(...),
                       pdf_file: UploadFile = None,
                       video_file: UploadFile = None):
    if not pdf_file or not video_file:
        raise HTTPException(status_code=400, detail="Both PDF and Video required")

    company_id = generate_uid(company_name)
    folder = f"{company_id}/"
    bucket = storage_client.bucket(BUCKET_NAME)

    # upload PDF
    pdf_blob = bucket.blob(folder + "pitch.pdf")
    pdf_blob.upload_from_file(pdf_file.file, content_type="application/pdf")

    # upload video
    vid_blob = bucket.blob(folder + "pitch.mp4")
    vid_blob.upload_from_file(video_file.file, content_type="video/mp4")

    # create company.json
    company_metadata = {"company_id": company_id, "company_name": company_name}
    json_blob = bucket.blob(folder + "company.json")
    json_blob.upload_from_string(json.dumps(company_metadata), content_type="application/json")

    return {
        "company_id": company_id,
        "gcs_folder": f"gs://{BUCKET_NAME}/{folder}",
        "status": "uploaded"
    }
