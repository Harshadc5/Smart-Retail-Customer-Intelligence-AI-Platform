# 🛍️ Smart Retail & Customer Intelligence AI Platform

```bash
uvicorn app.main:app --reload
```


An enterprise-grade, end-to-end Artificial Intelligence pipeline designed to modernize the retail experience. This project combines Computer Vision, Natural Language Processing, and a microservices architecture to automate store operations, monitor customer sentiment, and provide seamless customer support.

## 🚀 Key Features

* **Biometric Face Recognition:** Utilizes OpenCV Haar Cascades and LBPH (Local Binary Pattern Histograms) algorithms to instantly detect and verify VIP customers against a massive database of over 400 individuals.
* **Product Classification Engine:** A Convolutional Neural Network (CNN) trained on the full 60,000-image Fashion-MNIST dataset to automatically identify and categorize retail clothing items in real-time.
* **NLP Customer Sentiment Analysis:** Uses a fine-tuned Hugging Face `DistilBERT` transformer model to analyze incoming product reviews and automatically score customer sentiment.
* **Automated Support Chatbot:** An intent-based neural network trained with PyTorch that handles customer inquiries (refunds, hours, stock) dynamically.
* **Live Analytics Dashboard:** A real-time `Streamlit` frontend that allows managers to test the AI models live and monitor confidence drift.

## 🛠️ Technology Stack

* **Machine Learning / AI:** TensorFlow, PyTorch, OpenCV, Scikit-Learn, Hugging Face Transformers
* **Backend:** FastAPI, Uvicorn, Python
* **Frontend:** Streamlit
* **Environment:** Jupyter Notebooks, Docker

## ⚙️ How it Works (The Architecture)

The entire project can be generated and deployed from scratch using the included Jupyter Notebook (`Smart_Retail_Final_Project.ipynb`). When the notebook is executed, it:

1. Automatically downloads all massive datasets (including GitHub fallback fetching for Cloud environments like Colab).
2. Trains all Neural Networks (CNN, NLP, Chatbot) and saves the `.h5`, `.pkl`, and `.yml` models.
3. Automatically writes the complete FastAPI backend architecture into an `app/` directory.
4. Generates the frontend Streamlit dashboard.

## 💻 Running the Dashboard Locally

If you want to test the live API and dashboard on your local machine, open two terminals and run:

**Terminal 1 (Backend):**

```bash
uvicorn app.main:app --reload
```

**Terminal 2 (Frontend):**

```bash
streamlit run streamlit_app.py
```
