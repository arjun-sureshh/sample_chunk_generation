from fastapi import APIRouter, Query
from .data_loader import load_frames

router = APIRouter(prefix="/frames", tags=["Frames"])

@router.get("")
def get_frames(
    video_id: str = Query(None),
    phone: bool = Query(None),
    staff: bool = Query(None),
    crowd: bool = Query(None),
    limit: int = 50
):
    frames = load_frames()

    if video_id:
        frames = [f for f in frames if f["video_id"] == video_id]

    if phone is not None:
        frames = [f for f in frames if f["violations"]["phone"] == phone]

    if staff is not None:
        frames = [f for f in frames if f["violations"]["staff"] == staff]

    if crowd is not None:
        frames = [f for f in frames if f["violations"]["crowd"] == crowd]

    return {
        "total": len(frames),
        "data": frames[:limit]
    }
