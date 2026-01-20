import os
import cv2
from detector import detect_people
from analyzer import detect_staff_crowding
from drawer import draw_boxes
from utils import crop_person
from vlm_processor import analyze_frame

def vlm_worker(vlm_queue):
    print("[VLM] Worker started")

    while True:
        item = vlm_queue.get()
        if item is None:
            print("[VLM] Worker stopped")
            break

        frame_names = item["frame_names"]
        fps = {"fps": item["fps"]}

        for frame_path in frame_names:
            try:
                frame = cv2.imread(frame_path)
                persons = detect_people(frame)

                person_dir = frame_path.replace(".jpg", "_persons")
                os.makedirs(person_dir, exist_ok=True)

                # ---- VLM PER PERSON ----
                for p in persons:
                    crop_path = crop_person(
                        frame,
                        p["bbox"],
                        person_dir,
                        p["person_id"]
                    )

                    result = analyze_frame(crop_path)

                    # VERY IMPORTANT: parse JSON safely
                    p["is_staff"] = '"is_staff": true' in result.lower()
                    p["phone_visible"] = '"phone_visible": true' in result.lower()

                # ---- CROWD LOGIC (CODE, NOT VLM) ----
                crowded_ids = detect_staff_crowding(persons)

                # ---- DRAW LAST ----
                final_frame = draw_boxes(frame, persons, crowded_ids)
                out_path = frame_path.replace(".jpg", "_final.jpg")
                cv2.imwrite(out_path, final_frame)

                # ---- SAVE TEXT OUTPUT ----
                txt_path = frame_path.replace(".jpg", ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    for p in persons:
                        f.write(str(p) + "\n")
                    f.write("\n" + str(fps))

                print(f"[OK] Processed {frame_path}")

            except Exception as e:
                print(f"[VLM ERROR] {frame_path}: {e}")
