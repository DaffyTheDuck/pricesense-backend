from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.analyze import router as analyze_router
from routes.history import router as history_router
from routes.chat import router as chat_router

# Create the FastAPI app
# title and description show up in the auto-generated API docs
app = FastAPI(
    title="PriceSense AI",
    description="AI-powered promotion intelligence for retailers",
    version="1.0.0"
)

# ─── CORS CONFIGURATION ───────────────────────────────────────────
# This tells the backend which frontends are allowed to talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",        # local Next.js dev
        "https://*.vercel.app",         # any Vercel deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],                # allow GET, POST, PUT, DELETE etc.
    allow_headers=["*"],                # allow all headers
)

# ─── REGISTER ROUTES ──────────────────────────────────────────────
# Each router handles a feature area
# The prefix means every route inside gets that prefix
# So analyze.py's "/analyze" becomes "/api/analyze"
app.include_router(analyze_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


# ─── HEALTH CHECK ─────────────────────────────────────────────────
# A simple endpoint that returns "ok"
# Used by Railway to know the server is alive
# Also useful for you to test if the backend is running
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "PriceSense AI Backend",
        "version": "1.0.0"
    }


# ─── ROOT ─────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "PriceSense AI API",
        "docs": "/docs",        # FastAPI auto-generates docs here
        "health": "/health"
    }