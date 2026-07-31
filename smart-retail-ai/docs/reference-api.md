# API Reference

The unified REST and WebSocket endpoints for the Smart Retail AI Platform.

## Vision API (`app/routers/vision.py`)

### `POST /recognize-face`
Detects and identifies returning VIP customers from a static image.
- **Payload**: `multipart/form-data` (UploadFile `file`)
- **Returns**: `{"faces": ["Customer_1", "Unknown"]}`

### `POST /classify-product`
Classifies a retail apparel item using the custom CNN.
- **Payload**: `multipart/form-data` (UploadFile `file`)
- **Returns**: `{"product_class": "T-shirt/top", "confidence": 0.98}`

### `WS /ws/video`
Real-time biometric video streaming over WebSockets.
- **Payload**: Base64 encoded JPEG frames (string).
- **Returns**: `{"recognized": [...]}`

## NLP API (`app/routers/nlp.py`)

### `POST /analyze-sentiment`
Extracts emotional intent and confidence drift metrics from product reviews using HuggingFace DistilBERT.
- **Payload**: `{"text": "This jacket is amazing but slightly overpriced."}`
- **Returns**: `{"sentiment": "POSITIVE", "confidence": 0.94}`

## Chatbot API (`app/routers/chatbot.py`)

### `POST /chatbot`
Provides automated retail FAQ responses using TF-IDF semantic matching.
- **Payload**: `{"query": "What time do you close on Sunday?"}`
- **Returns**: `{"response": "Our store hours are 9 AM to 9 PM daily."}`

### `POST /chatbot-rate`
Logs user satisfaction metrics for A/B testing.
- **Payload**: `{"strategy": "A", "rating": 5}` (Rating must be 1-5).
- **Returns**: `{"status": "logged"}`

## Related
- See [How to stream real-time facial recognition via WebSockets](./howto-websocket-video-stream.md).
- See [Why we use DistilBERT](./explanation-distilbert.md).
