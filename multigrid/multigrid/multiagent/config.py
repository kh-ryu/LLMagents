from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class AgentConfigModel(BaseModel):
    sys_prompt_file: str = Field(..., description="the system prompt file of the agent.", alias="sys_prompt")
    temperature: float = Field(..., le=0.0, ge=1.0, description="llm's temperature.")
    top_p: float = Field(..., le=0.0, ge=1.0, description="llm's top-p")
        