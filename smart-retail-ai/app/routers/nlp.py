from fastapi import APIRouter, Request  # type: ignore
from pydantic import BaseModel  # type: ignore

router = APIRouter()

class TextRequest(BaseModel):
    text: str

@router.post('/analyze-sentiment')
def analyze_sentiment(request: Request, req: TextRequest):
    pipeline = request.app.state.pipeline
    if not pipeline: return {'error': 'Pipeline not loaded.'}
    result, confidence = pipeline.predict_sentiment(req.text)
    return {'status': 'success', 'sentiment': result, 'confidence': float(confidence)}

