from fastapi import APIRouter, Request, UploadFile, File  # type: ignore
import cv2  # type: ignore
import numpy as np
import base64  # type: ignore

router = APIRouter()

@router.post('/recognize-face')
async def recognize_face(request: Request, file: UploadFile = File(...)):
    pipeline = request.app.state.pipeline
    if not pipeline: return {'error': 'Pipeline not loaded.'}
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label, confidence, name, cropped_face = pipeline.recognize_face(gray)

    response = {'status': 'success', 'face_id': int(label), 'face_name': name, 'confidence': float(confidence)}

    if cropped_face is not None:
        _, buffer = cv2.imencode('.jpg', cropped_face)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        response['cropped_face_b64'] = b64_str

    return response

@router.post('/classify-product')
async def classify_product(request: Request, file: UploadFile = File(...)):
    pipeline = request.app.state.pipeline
    if not pipeline: return {'error': 'Pipeline not loaded.'}
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(gray, (28, 28))
    prediction = pipeline.product_classifier.predict(resized.reshape(1, 28, 28, 1), verbose=0)
    cat_idx = int(np.argmax(prediction[0]))
    conf = float(prediction[0][cat_idx])
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return {'status': 'success', 'category': classes[cat_idx], 'confidence': conf}

