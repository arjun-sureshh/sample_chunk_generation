from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.frames_api import router as frames_router
from api.videos_api import router as videos_router

app = FastAPI(title="Video Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(frames_router)
app.include_router(videos_router)
