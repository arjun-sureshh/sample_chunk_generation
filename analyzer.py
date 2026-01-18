# analyzer.py
import math

def center(b):
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def detect_staff_crowding(persons, threshold=120):
    staff = [p for p in persons if p["is_staff"]]
    crowded = set()

    for i in range(len(staff)):
        for j in range(i+1, len(staff)):
            c1 = center(staff[i]["bbox"])
            c2 = center(staff[j]["bbox"])
            dist = math.dist(c1, c2)
            if dist < threshold:
                crowded.add(staff[i]["person_id"])
                crowded.add(staff[j]["person_id"])

    return crowded
