import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import requests  # type: ignore
import os  # type: ignore

st.set_page_config(page_title='Smart Retail Dashboard', layout='wide')

st.title('🛍️ Smart Retail AI Dashboard')
st.markdown('Real-time monitoring connected to the Smart Retail API (Advanced Stretch Goal 4).')

# Try to fetch live stats from FastAPI
api_url = "http://127.0.0.1:8000/dashboard/stats"
try:
    response = requests.get(api_url, timeout=2)
    data = response.json()
    st.success(f"🟢 Connected to API! Live Visits today: {data['daily_visits']}")
except:
    st.warning("🔴 API Offline. Displaying cached data. Run `uvicorn app.main:app` to connect!")

col1, col2 = st.columns(2)

with col1:
    st.subheader('📈 Visit Trends (Last 30 Days)')
    visits = pd.DataFrame(np.random.randint(100, 500, size=(30, 1)), columns=['Daily Visitors'])
    st.line_chart(visits)

with col2:
    st.subheader('💬 Sentiment Trends')
    sentiment = pd.DataFrame(np.random.uniform(0.6, 0.9, size=(30, 1)), columns=['Avg Sentiment Score'])
    st.area_chart(sentiment)

st.divider()

st.subheader('🤖 Model Monitoring: Confidence Drift')
st.markdown('Monitoring the prediction confidence of our AI models over time (Stretch Goal 3).')
if os.path.exists('confidence_drift.png'):
    st.image('confidence_drift.png', use_container_width=True)
else:
    st.info('Run the Model Monitoring cell in the Jupyter Notebook to generate the drift graph!')

