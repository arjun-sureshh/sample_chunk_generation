import os
import cv2
from ultralytics import YOLO
from config import CONFIG

# Load the model once with config
model = YOLO(CONFIG["yolo"]["model_path"])

def yolo_worker(yolo_queue):
    
    print("[YOLO] Worker started. Waiting for frames...")
    base_output_dir = "result/detections"
    os.makedirs(base_output_dir, exist_ok=True)

    while True:
        item = yolo_queue.get()
        if item is None:
            print("[YOLO] Received stop signal")
            break

        chunk_id = item["chunk_id"]
        frame_names = item["frame_names"]
        
        # Create chunk-specific output directory
        chunk_output_dir = os.path.join(base_output_dir, chunk_id)
        os.makedirs(chunk_output_dir, exist_ok=True)
        
        print(f"\n[YOLO] Processing chunk: {chunk_id} with {len(frame_names)} frames")
        print(f"[YOLO] Output directory: {chunk_output_dir}")

        for idx, img_path in enumerate(frame_names):
            try:
                print(f"[YOLO] Processing frame {idx+1}/{len(frame_names)}: {img_path}")
                
                # Check if file exists
                if not os.path.exists(img_path):
                    print(f"[YOLO ERROR] File not found: {img_path}")
                    continue
                
                # Run YOLO inference using config values
                results = model.predict(
                    source=img_path,
                    classes=CONFIG["yolo"]["classes"],
                    conf=CONFIG["yolo"]["confidence"],
                    verbose=False
                )

                for result in results:
                    # 1. Save Coordinates (.txt) in chunk folder
                    txt_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
                    txt_path = os.path.join(chunk_output_dir, txt_name)
                    result.save_txt(txt_path)

                    # 2. Add "Total Count" to the image
                    annotated_frame = result.plot()
                    person_count = len(result.boxes)
                    
                    cv2.putText(
                        annotated_frame,
                        f"Total Persons: {person_count}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3
                    )

                    # 3. Save the annotated image in chunk folder
                    save_path = os.path.join(chunk_output_dir, "annotated_" + os.path.basename(img_path))
                    cv2.imwrite(save_path, annotated_frame)
                    
                    print(f"[YOLO] ✓ Saved: {save_path} (Persons: {person_count})")
                    
            except Exception as e:
                print(f"[YOLO ERROR] Failed to process {img_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        yolo_queue.task_done()
        print(f"[YOLO] Completed chunk: {chunk_id}\n")
    
    print("[YOLO] Worker stopped")