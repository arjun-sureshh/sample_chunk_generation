from fastapi import APIRouter
from .data_loader import load_frames

router = APIRouter(prefix="/videos", tags=["Videos"])

@router.get("/summary")
def video_summary():
    frames = load_frames()
    videos = {}

    for f in frames:
        vid = f["video_id"]

        if vid not in videos:
            videos[vid] = {
                "video_id": vid,
                "total_frames": 0,
                "violations": {
                    "phone": 0,
                    "staff": 0,
                    "crowd": 0
                }
            }

        videos[vid]["total_frames"] += 1

        for v in videos[vid]["violations"]:
            if f["violations"].get(v):
                videos[vid]["violations"][v] += 1

    return list(videos.values())


@router.get("/{video_id}/violations")
def video_violations(video_id: str):
    frames = load_frames()

    result = {
        "phone": [],
        "staff": [],
        "crowd": []
    }

    for f in frames:
        if f["video_id"] != video_id:
            continue
        for k in result:
            if f["violations"].get(k):
                result[k].append(f)

    return {
        "video_id": video_id,
        "violations": {
            k: {
                "count": len(v),
                "frames": v
            } for k, v in result.items()
        }
    }
