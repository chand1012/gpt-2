from typing import List, Optional
from pydantic import BaseModel

# need to get these working

class ResponseModel(BaseModel):
    prompt: str # Echoed prompt
    responses: List[str]

class RequestModel(BaseModel):
    prompt: str
    nsamples: Optional[int] = 1
    model_name: Optional[str] = '124M'
    batch_size: Optional[int] = 1
    seed: Optional[float] = None
    length: Optional[int] = None
    top_k: Optional[int] = 40
    temperature: Optional[float] = 1.0