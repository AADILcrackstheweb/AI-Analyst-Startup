FROM python:3.11-slim

WORKDIR /app

# Avoid debconf errors during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port 8080 (Cloud Run expects this)
EXPOSE 8080

# Run Streamlit app
CMD ["streamlit", "run", "ui.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false"]
