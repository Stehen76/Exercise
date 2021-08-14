import summarizer_and_keywords_extractor
import uvicorn
import os
from fastapi import FastAPI

# ML Pkg
# import joblib,os
import text2emotion as te
from summarizer_and_keywords_extractor import summarize_and_keywords_en
from summarizer_and_keywords_extractor import summarize_and_keywords_es

# init app
app = FastAPI()

# Routes
@app.get('/')
async def index():
    return {"text": "Hello API Muster"}

@app.get('/items/{name}')
async def get_items(name):
    return {"name": name}

# ML Aspect
@app.get('/predict_emotion/{text}')
async def predict_emotion(text):
    emotion = te.get_emotion(text)
    return {'text': text, 'emotion': emotion}

@app.get('/summaries_and_keywords_en/{text}')
async def summarize_plus_keywords_en(text):
    x = summarize_and_keywords_en(text)
    return x


@app.get('/summaries_and_keywords_es/{text}')
async def summarize_plus_keywords_es(text):
    x = summarize_and_keywords_es(text)
    return x



if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)