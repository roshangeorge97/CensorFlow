from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict

# Model definitions
class ModerateTextRequest(BaseModel):
    text: str

class ModerateBatchRequest(BaseModel):
    texts: List[str]

class SuggestTextRequest(BaseModel):
    text: str
    moderation_result: Dict[str, float]

class TaskRequest(BaseModel):
    taskId: str
