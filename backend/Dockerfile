# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy dependencies first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend files
COPY . .

# Run the app with uvicorn
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]
