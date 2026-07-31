# How to run Chatbot A/B Tests

Route live customer traffic between two different chatbot interaction strategies (Formal vs. Casual) and mathematically deduce the optimal brand voice.

## Prerequisites

- The FastAPI server must be running.

## Steps

1. Send simulated interactions to the `/chatbot-rate` endpoint:

   ```bash
   curl -X POST "http://127.0.0.1:8000/chatbot-rate" \
        -H "Content-Type: application/json" \
        -d '{"strategy": "A", "rating": 5}'
   ```
   *Strategy A represents the Formal tone.*

2. Send a rating for the alternative strategy:

   ```bash
   curl -X POST "http://127.0.0.1:8000/chatbot-rate" \
        -H "Content-Type: application/json" \
        -d '{"strategy": "B", "rating": 2}'
   ```
   *Strategy B represents the Casual tone.*

3. Analyze the results visually using the dashboard:

   ```bash
   streamlit run streamlit_app.py
   ```

## Verification

Open the Streamlit dashboard and scroll to the **A/B Testing** section. You will see a live bar chart comparing the average satisfaction ratings of Strategy A versus Strategy B.
