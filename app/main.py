from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import query, ingest

app = FastAPI(
    title="SERA AI API",
    description="Backend API for the Self Evolving RAG Assistant (SERA)",
    version="1.0.0"
)

# Configure CORS so the frontend can hit these endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers
app.include_router(query.router)
app.include_router(ingest.router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SERA API is running"}
