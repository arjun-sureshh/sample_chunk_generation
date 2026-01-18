# drawer.py
import cv2

def draw_boxes(frame, persons, crowded_ids):
    for p in persons:
        x1, y1, x2, y2 = p["bbox"]

        if p["is_staff"]:
            color = (0, 255, 0)
            label = f"STAFF {p['person_id']}"
            if p["phone_visible"]:
                color = (0, 0, 255)
                label += " PHONE"
            if p["person_id"] in crowded_ids:
                color = (0, 255, 255)
                label += " CROWD"
        else:
            color = (255, 0, 0)
            label = f"CUSTOMER {p['person_id']}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame
