"""Compatibility wrapper for the FastAPI service."""

from app.api.server import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=True)
