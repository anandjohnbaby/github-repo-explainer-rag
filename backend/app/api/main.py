from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pathlib import Path

from app.api.routes import router as api_router

# ---------------- PATH SETUP ----------------
# Project root → github-repo-chatbot/
BASE_DIR = Path(__file__).resolve().parents[3]
FRONTEND_DIR = BASE_DIR / "frontend"
# --------------------------------------------

app = FastAPI(title="GitHub Repo Chatbot (RAG)")

# ---------------- CORS ----------------
# Allows frontend (browser) to call backend APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # safe for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------

# ---------------- STATIC FILES ----------------
# Serve frontend static assets
app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)
# ----------------------------------------------

# ---------------- TEMPLATES ----------------
templates = Jinja2Templates(directory=FRONTEND_DIR)
# -------------------------------------------

# ---------------- API ROUTES ----------------
app.include_router(api_router)
# --------------------------------------------

# ---------------- FRONTEND ENTRY ----------------
@app.get("/")
def serve_ui(request: Request):
    # Serve frontend UI
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )
# -----------------------------------------------
