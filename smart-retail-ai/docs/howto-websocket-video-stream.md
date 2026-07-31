# How to stream real-time facial recognition via WebSockets

Connect to the API using WebSockets to achieve 30FPS real-time facial recognition, bypassing the extreme latency overhead of standard HTTP polling.

## Prerequisites

- The FastAPI server must be running (`uvicorn app.main:app --reload`).
- A client script capable of WebSocket connections (e.g., Python `websockets` library).

## Steps

1. Install the websockets library in your client environment:

   ```bash
   pip install websockets opencv-python
   ```

2. Connect to the endpoint and stream base64-encoded frames. Use this Python snippet:

   ```python
   import asyncio
   import websockets
   import cv2
   import base64
   
   async def stream_video():
       uri = "ws://127.0.0.1:8000/ws/video"
       async with websockets.connect(uri) as websocket:
           cap = cv2.VideoCapture(0)
           while cap.isOpened():
               ret, frame = cap.read()
               if not ret: break
               _, buffer = cv2.imencode('.jpg', frame)
               encoded_frame = base64.b64encode(buffer).decode('utf-8')
               
               await websocket.send(encoded_frame)
               response = await websocket.recv()
               print(f"Server recognized: {response}")
   
   asyncio.run(stream_video())
   ```
   *This captures your webcam, encodes it, and prints the recognized VIP customer names.*

## Verification

If successful, your terminal will rapidly print JSON responses like `{"recognized": ["Customer A"]}` at a high frame rate.

## Troubleshooting

- **Connection Refused**: Ensure FastAPI is running on port 8000.
- **WebSocketDisconnect**: The server dropped the connection, likely due to a malformed base64 string. Ensure you decode the byte buffer to a `utf-8` string before sending.
