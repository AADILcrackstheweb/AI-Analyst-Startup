import streamlit as st
import requests

BACKEND_URL = "https://backend-service-196789399336.asia-south1.run.app/upload"

st.set_page_config(page_title="Startup Ingestion", page_icon="🚀", layout="centered")
st.title("🚀 Startup Ingestion Portal")

st.markdown("Fill in your startup details and upload your pitch materials.")

with st.form("startup_form", clear_on_submit=False):
    company_name = st.text_input("Startup Name")
    pdf_file = st.file_uploader("Upload Pitch PDF", type=["pdf"])
    video_file = st.file_uploader("Upload Pitch Video", type=["mp4", "mov", "avi"])
    submitted = st.form_submit_button("Submit")

if submitted:
    if not company_name or not pdf_file or not video_file:
        st.error("Please provide company name, PDF, and video.")
    else:
        with st.spinner("Uploading to backend..."):
            files = {
                "pdf_file": (pdf_file.name, pdf_file.getvalue(), pdf_file.type),
                "video_file": (video_file.name, video_file.getvalue(), video_file.type)
            }
            data = {"company_name": company_name}

            try:
                response = requests.post(BACKEND_URL, data=data, files=files, timeout=120)
                if response.status_code == 200:
                    st.success("✅ Files uploaded successfully!")
                    st.json(response.json())
                else:
                    st.error(f"❌ Error {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
