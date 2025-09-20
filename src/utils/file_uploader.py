# src/utils/file_handler.py

import os
import aiofiles
import hashlib
from typing import Optional, List
from datetime import datetime
from pathlib import Path
from fastapi import UploadFile
import magic
from google.cloud import storage

class FileHandler:
    """Handle file uploads and storage"""
    
    def __init__(self):
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        
        # Initialize Google Cloud Storage client
        try:
            self.gcs_client = storage.Client()
            self.bucket_name = os.getenv("GCS_BUCKET_NAME", "due-diligence-documents")
        except Exception:
            self.gcs_client = None
    
    async def save_uploaded_file(self, file: UploadFile, analysis_id: str) -> str:
        """Save uploaded file locally and optionally to GCS"""
        
        # Validate file
        await self._validate_file(file)
        
        # Generate safe filename
        safe_filename = self._generate_safe_filename(file.filename, analysis_id)
        
        # Create analysis directory
        analysis_dir = self.upload_dir / analysis_id
        analysis_dir.mkdir(exist_ok=True)
        
        # Save file locally
        local_path = analysis_dir / safe_filename
        
        async with aiofiles.open(local_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Upload to GCS if available
        gcs_path = None
        if self.gcs_client:
            try:
                gcs_path = await self._upload_to_gcs(local_path, analysis_id, safe_filename)
            except Exception as e:
                print(f"Failed to upload to GCS: {e}")
        
        return str(local_path)
    
    async def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file"""
        
        # Check file size (max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if file.size > max_size:
            raise ValueError(f"File too large. Maximum size is {max_size / 1024 / 1024}MB")
        
        # Check file type
        allowed_types = {
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/plain',
            'application/vnd.ms-powerpoint',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        }
        
        if file.content_type not in allowed_types:
            raise ValueError(f"File type {file.content_type} not allowed")
        
        # Check filename
        if not file.filename or len(file.filename) > 255:
            raise ValueError("Invalid filename")
    
    def _generate_safe_filename(self, original_filename: str, analysis_id: str) -> str:
        """Generate safe filename"""
        
        # Get file extension
        file_ext = Path(original_filename).suffix.lower()
        
        # Create hash of original filename + timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{original_filename}_{timestamp}_{analysis_id}"
        file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        # Clean original name (remove special characters)
        clean_name = "".join(c for c in Path(original_filename).stem if c.isalnum() or c in ('-', '_'))
        clean_name = clean_name[:50]  # Limit length
        
        return f"{clean_name}_{file_hash}{file_ext}"
    
    async def _upload_to_gcs(self, local_path: Path, analysis_id: str, filename: str) -> str:
        """Upload file to Google Cloud Storage"""
        
        bucket = self.gcs_client.bucket(self.bucket_name)
        blob_name = f"analyses/{analysis_id}/{filename}"
        blob = bucket.blob(blob_name)
        
        # Upload file
        blob.upload_from_filename(str(local_path))
        
        return f"gs://{self.bucket_name}/{blob_name}"
    
    async def get_file_content(self, file_path: str) -> bytes:
        """Get file content"""
        
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file"""
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception:
            return False
    
    async def cleanup_analysis_files(self, analysis_id: str) -> None:
        """Clean up files for an analysis"""
        
        analysis_dir = self.upload_dir / analysis_id
        
        if analysis_dir.exists():
            # Delete all files in the directory
            for file_path in analysis_dir.iterdir():
                if file_path.is_file():
                    await self.delete_file(str(file_path))
            
            # Remove directory
            try:
                analysis_dir.rmdir()
            except OSError:
                pass  # Directory not empty
    
    def get_file_info(self, file_path: str) -> dict:
        """Get file information"""
        
        if not os.path.exists(file_path):
            return None
        
        stat = os.stat(file_path)
        
        return {
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "extension": Path(file_path).suffix.lower()
        }