from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import base64
import joblib

# Load your trained sentiment model
model = joblib.load("senti.ykl")

# Sentiment → Image path mapping
sentiment_to_image = {
    "Positive": "static/pos.png",
    "Neutral": "static/neu.png",
    "Negative": "static/neg.png",
    "Irrelevant": "static/irrelevant.png"
}

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def form_post(request: Request, text: str = Form(...)):
    prediction = model.predict([text])[0]
    image_path = sentiment_to_image[prediction]

    # Convert image to base64 for embedding
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "text": text,
        "prediction": prediction,
        "image_data": encoded_image
    })
