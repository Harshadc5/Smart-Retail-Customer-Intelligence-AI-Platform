# Getting Started with Smart Retail AI

You will build and deploy a working, enterprise-grade AI microservice for a smart retail environment. By the end of this tutorial, you will understand how to generate the API, start the web server, and view live analytics.

## What you'll need

- Python 3.11 installed
- A web browser (for the Streamlit dashboard)

## Step 1: Install Dependencies

First, we need to ensure all required libraries (like FastAPI, TensorFlow, OpenCV, and PyTorch) are installed.

```bash
pip install -r requirements.txt
```
This installs the machine learning frameworks and the web server dependencies.

## Step 2: Build the Architecture

We use a "Notebook-as-Generator" pattern. You must run the builder notebook to generate the models and the `app/` folder structure.

```bash
jupyter nbconvert --to script Smart_Retail_Final_Project.ipynb
python Smart_Retail_Final_Project.py
```
This trains the CNN, downloads the DistilBERT weights, and serializes everything into the `app/models/` directory.

## Step 3: Start the Backend Server

Start the FastAPI ASGI server using Uvicorn.

```bash
uvicorn app.main:app --reload
```
You should see `Uvicorn running on http://127.0.0.1:8000`.

## Step 4: Open the Live Dashboard

In a new terminal window, start the Streamlit frontend.

```bash
streamlit run streamlit_app.py
```

## What you built

You now have a fully operational AI backend exposing REST and WebSocket endpoints for Vision and NLP, paired with a real-time analytics dashboard! 
- Dive into the [API Reference](./reference-api.md) to see the endpoints.
- Learn [How to run A/B Tests](./howto-run-ab-tests.md).
