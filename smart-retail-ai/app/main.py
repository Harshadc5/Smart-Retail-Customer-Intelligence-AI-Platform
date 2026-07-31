from fastapi import FastAPI, WebSocket  # type: ignore
import logging
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
import os  # type: ignore
from app.services.pipeline import SmartRetailPipeline  # type: ignore
from app.routers import vision, nlp, chatbot  # type: ignore

app = FastAPI(title='Smart Retail AI API')

logging.info("Loading Smart Retail AI Pipeline...")
try:
    if os.path.exists('app/models/product_classifier.h5'):
        app.state.pipeline = SmartRetailPipeline()
    else:
        app.state.pipeline = None
except Exception as e:
    logging.error(f"Warning: Pipeline not loaded. Error: {e}")
    app.state.pipeline = None

app.include_router(vision.router)
app.include_router(nlp.router)
app.include_router(chatbot.router)

@app.get('/dashboard/stats')
def dashboard_stats():
    import random
    return {'daily_visits': random.randint(300, 500), 'average_sentiment': round(random.uniform(0.7, 0.9), 2)}

@app.websocket('/ws/video')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    import cv2, numpy as np, base64, json
    while True:
        data = await websocket.receive_text()
        try:
            nparr = np.frombuffer(base64.b64decode(data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            pipeline = websocket.app.state.pipeline
            if pipeline:
                label, conf, name, _ = pipeline.recognize_face(gray)
                await websocket.send_text(json.dumps({'recognized': [name]}))
            else:
                await websocket.send_text(json.dumps({'recognized': ['Unknown (Pipeline offline)']}))
        except Exception:
            await websocket.send_text(json.dumps({'recognized': ['Error decoding']}))

@app.get('/')
def read_root(): return {'message': 'Smart Retail AI API is running.'}

