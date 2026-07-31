from fastapi import APIRouter, Request  # type: ignore
from pydantic import BaseModel  # type: ignore

router = APIRouter()

class TextRequest(BaseModel):
    text: str

@router.post('/chatbot')
def chatbot(request: Request, req: TextRequest):
    pipeline = request.app.state.pipeline
    if not pipeline: return {'error': 'Pipeline not loaded.'}
    reply = pipeline.chat_response(req.text)
    return {'status': 'success', 'reply': reply}

