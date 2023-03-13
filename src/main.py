from utils.metadata import *
from fastapi import FastAPI, Request
from pathlib import Path
from fastapi.templating import Jinja2Templates
from utils.predictor import url_predict
from fastapi.responses import RedirectResponse, HTMLResponse
import uvicorn


BASE_PATH = Path(__file__).resolve().parent
templates = Jinja2Templates(directory = str(BASE_PATH / "templates"))

app = FastAPI(
    title = "PokéRecognition",
    description = "Free Pokémon Recognition API",
    version = "0.0.1",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    openapi_tags = tags_metadata
)

# Error handlers
@app.exception_handler(403)
async def exception_403(request: Request):
    return {
        "success": False,
        "error": request
    }

@app.exception_handler(404)
async def exception_404(request: Request):
    return RedirectResponse("/")

# Docs endpoints
@app.get("/endpoints",  tags = ["API Endpoints"])
async def get_all_urls_from_request(request: Request):
    endpoints = [
        {
            "path": app.route.path,
            "name": app.route.name
        } for app.route in request.app.routes
    ]
    return {
        "success": True,
        "endpoints": endpoints
    }

# Home page
@app.get("/", response_class = HTMLResponse, tags = ["Home Page"])
async def home_page(request: Request) -> dict:
    return templates.TemplateResponse(
        "home.html",
        {"request": request}
    )

# Predict request
@app.get("/api/predict/url", response_class = HTMLResponse, tags = ["Predict A Pokémon Name"])
async def predict_pokémon_frontend(request: Request, url: str):
    # Predict pokemon
    prediction = url_predict(url)

    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "url": url,
            "pokemon_name": prediction.tolist()
        }
    )

@app.post("/api/predict/url", tags = ["Predict A Pokémon Name"])
async def predict_pokémon_url(url):
    prediction = url_predict(url)

    return {
        "success": True,
        "prediction": prediction.tolist()
    }

# Start program
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host = "127.0.0.1",
        port = 5000,
        log_level = "info"
    )
