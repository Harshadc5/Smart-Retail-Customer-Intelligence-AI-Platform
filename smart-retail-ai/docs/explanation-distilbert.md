# Deep Contextual Sentiment Analysis (DistilBERT)

The NLP module utilizes a HuggingFace Transformer (`DistilBertForSequenceClassification`) instead of legacy algorithms to analyze customer product reviews.

## The problem

Legacy algorithms like TF-IDF or simple Naive Bayes struggle with semantic context and sarcasm. A review stating *"Not exactly what I would call a bad product"* contains the words "not" and "bad", which legacy models heavily penalize, resulting in a false "Negative" classification for a genuinely neutral/positive review.

## The approach

Transformers utilize multi-headed attention mechanisms to read text bidirectionally, understanding the context of words based on their surroundings. By fine-tuning DistilBERT, the system understands retail nuance.

Furthermore, we extract mathematical confidence using PyTorch Softmax on the model logits:

```python
import torch
probs = torch.nn.functional.softmax(outputs.logits, dim=1)
confidence = probs.max().item()
```

This yields a float (`0.0` to `1.0`), empowering the business to set dynamic thresholds (e.g., routing angry customers to human agents if negative sentiment confidence exceeds `0.92`).

## Trade-offs

- **Speed & Memory**: DistilBERT is exponentially heavier than TF-IDF. It requires ~250MB of system memory just to load the tensors, and CPU inference takes ~50ms compared to TF-IDF's ~2ms.
- **Solution**: We mitigate this by loading the model into system memory exactly *once* during FastAPI's startup lifecycle (`app.state.pipeline`), preventing OOM crashes on concurrent requests.
