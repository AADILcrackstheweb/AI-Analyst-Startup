# src/tools/base_tool.py

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type
from datetime import datetime

from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, ValidationError

from config.settings import get_settings
from src.data.models import BaseResponse, ConfidenceScore

class BaseDueDiligenceTool(BaseTool, ABC):
    """Base class for all due diligence tools with Pydantic integration"""
    
    # Define input and output models for each tool
    input_model: Type[BaseModel] = None
    output_model: Type[BaseModel] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = get_settings()
        self._execution_time: Optional[float] = None
    
    def validate_input(self, **kwargs) -> BaseModel:
        """Validate input using Pydantic model"""
        if not self.input_model:
            raise NotImplementedError("input_model must be defined")
        
        try:
            return self.input_model(**kwargs)
        except ValidationError as e:
            print(f"Input validation failed: {e}")
    
    def validate_output(self, result: Dict[str, Any]) -> BaseModel:
        """Validate output using Pydantic model"""
        if not self.output_model:
            return result
        
        try:
            return self.output_model(**result)
        except ValidationError as e:
            print(f"Output validation failed: {e}")
            return result
    
    @abstractmethod
    async def _execute(self, validated_input: BaseModel) -> Dict[str, Any]:
        """Execute the tool's main functionality with validated input"""
        pass
    

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Run the tool with Pydantic validation"""
        try:
            # Validate input
            validated_input = self.validate_input(**kwargs)
            
            # Execute tool logic (run async in sync context)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._execute(validated_input))
            finally:
                loop.close()
            
            # Validate output
            validated_output = self.validate_output(result)
            
            # Return JSON string
            if isinstance(validated_output, BaseModel):
                return validated_output.json(indent=2)
            else:
                return json.dumps(result, indent=2, default=str)
                
       
        except Exception as e:
            print(f"Failed to execute {self.name}: {str(e)}")
    
    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format the output for consistency"""
        return {
            "tool_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "metadata": {
                "execution_time": getattr(self, '_execution_time', None),
                "confidence_score": result.get("confidence", {}).get("score", 0.0) if isinstance(result.get("confidence"), dict) else 0.0
            }
        }
    
    def _calculate_confidence_score(self, data_quality: float, result_completeness: float, source_reliability: float = 1.0) -> ConfidenceScore:
        """Calculate confidence score based on multiple factors"""
        # Weighted average of factors
        score = (data_quality * 0.4 + result_completeness * 0.4 + source_reliability * 0.2)
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        
        if score >= 0.7:
            level = "high"
        elif score >= 0.4:
            level = "medium"
        else:
            level = "low"
        
        return ConfidenceScore(score=score, level=level)