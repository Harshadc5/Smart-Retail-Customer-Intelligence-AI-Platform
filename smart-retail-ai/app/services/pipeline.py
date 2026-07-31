import logging
import joblib  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
import cv2  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from transformers import DistilBertTokenizer  # type: ignore
import random

class SmartRetailPipeline:
    def __init__(self):
        logging.info('Loading models...')
        try:
            self.product_classifier = load_model('app/models/product_classifier.h5')
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.face_recognizer.read('app/models/face_db.yml')
            self.face_names = joblib.load('app/models/face_db.pkl')
            import os
            cascade_path = os.path.abspath('app/models/haarcascade_frontalface_default.xml')
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.chatbot_model = joblib.load('app/models/chatbot_model.pkl')
            from transformers import DistilBertForSequenceClassification
            self.sentiment_model = DistilBertForSequenceClassification.from_pretrained('app/models/sentiment_model_distilbert')
            self.sentiment_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        except Exception as e:
            logging.error("Error loading models: %s", e)

    def recognize_face(self, gray_img):
        # returns label, confidence
        faces = self.face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return -1, 0.0, "No face detected", None

        (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
        face_roi = gray_img[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (37, 50))

        label, conf = self.face_recognizer.predict(face_resized)
        name = self.face_names[label] if label < len(self.face_names) else f"Person_{label}"
        return label, conf, name, face_resized

    def predict_sentiment(self, text):
        inputs = self.sentiment_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        return pred_class, confidence

    def chat_response(self, text):
        vec = self.chatbot_model['vectorizer'].transform([text])
        tag = self.chatbot_model['classifier'].predict(vec)[0]
        for intent in self.chatbot_model['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
        return 'I dont understand.'

