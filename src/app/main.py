from fastapi import FastAPI
from pydantic import BaseModel
from textblob import TextBlob

app = FastAPI(title="Sentiment API")

class TextData(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    polarity: float
    sentiment: str

@app.post("/predict", response_model=dict)
async def predict_sentiment(data: TextData):
    blob = TextBlob(data.text)
    polarity = blob.sentiment.polarity

    if polarity >= 0:
        sentiment = "положительный"
    elif polarity < 0:
        sentiment = "отрицательный"

    return {"polarity": polarity, "sentiment": sentiment}
