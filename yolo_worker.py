import os
import cv2
from ultralytics import YOLO
from config import CONFIG, ZONE
from zone_counter import scale_zone_to_image, count_people_in_zone

# Load the model once
model = YOLO(CONFIG["yolo"]["model_path"])

def yolo_worker(yolo_queue):
    
    print("[YOLO] Worker started - IOU-based Zone Counting")
    print(f"[YOLO] IOU threshold: {ZONE['iou_threshold']}")
    
    base_output_dir = "result/detections"
    os.makedirs(base_output_dir, exist_ok=True)

    while True:
        item = yolo_queue.get()
        if item is None:
            print("[YOLO] Received stop signal")
            break

        chunk_id = item["chunk_id"]
        frame_names = item["frame_names"]
        
        chunk_output_dir = os.path.join(base_output_dir, chunk_id)
        os.makedirs(chunk_output_dir, exist_ok=True)
        
        print(f"\n[YOLO] Processing chunk: {chunk_id} with {len(frame_names)} frames")

        for idx, img_path in enumerate(frame_names):
            try:
                print(f"[YOLO] Processing frame {idx+1}/{len(frame_names)}: {img_path}")
                
                if not os.path.exists(img_path):
                    print(f"[YOLO ERROR] File not found: {img_path}")
                    continue
                
                # Read image
                original_image = cv2.imread(img_path)
                image_height, image_width = original_image.shape[:2]
                
                # Scale zone to image size
                zone_box = scale_zone_to_image(ZONE, image_width, image_height)
                
                # Run YOLO detection
                results = model.predict(
                    source=img_path,
                    classes=CONFIG["yolo"]["classes"],
                    conf=CONFIG["yolo"]["confidence"],
                    verbose=False
                )

                for result in results:
                    # Extract all detected boxes and confidences
                    all_boxes = []
                    all_confs = []
                    
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        all_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                        all_confs.append(conf)
                    
                    total_detected = len(all_boxes)
                    
                    # Count people in zone using IOU
                    count_in_zone, boxes_in_zone, iou_values = count_people_in_zone(
                        all_boxes, 
                        zone_box, 
                        ZONE['iou_threshold']
                    )
                    
                    # Start with clean image
                    annotated_frame = original_image.copy()
                    
                    # Draw ONLY bounding boxes for people in zone
                    for i, box in enumerate(all_boxes):
                        if box not in boxes_in_zone:
                            continue
                        
                        # Get IOU value for this box
                        box_index = boxes_in_zone.index(box)
                        iou_value = iou_values[box_index]
                        
                        # Draw blue bounding box
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1), (x2, y2),
                            (255, 0, 0),  # Blue
                            2
                        )
                        
                        # Create label with confidence
                        label = f"person {all_confs[i]:.2f}"
                        (label_w, label_h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        
                        # Draw label background
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1 - label_h - 5),
                            (x1 + label_w, y1),
                            (255, 0, 0),
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )
                        
                        # Draw IOU value below the box
                        iou_label = f"IOU: {iou_value:.3f}"
                        (iou_w, iou_h), _ = cv2.getTextSize(
                            iou_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                        )
                        
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y2),
                            (x1 + iou_w, y2 + iou_h + 5),
                            (255, 0, 0),
                            -1
                        )
                        
                        cv2.putText(
                            annotated_frame,
                            iou_label,
                            (x1, y2 + iou_h),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1
                        )
                    
                    # Draw RED zone rectangle
                    zone_color = tuple(ZONE.get('color', [0, 0, 255]))
                    zone_thickness = ZONE.get('thickness', 3)
                    
                    cv2.rectangle(
                        annotated_frame,
                        (zone_box[0], zone_box[1]),
                        (zone_box[2], zone_box[3]),
                        zone_color,
                        zone_thickness
                    )
                    
                    # Draw ONLY GREEN count text (NO threshold label)
                    count_text = f"Total Persons: {count_in_zone}"
                    
                    cv2.putText(
                        annotated_frame,
                        count_text,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3
                    )
                    
                    # Save annotated image
                    save_path = os.path.join(chunk_output_dir, "annotated_" + os.path.basename(img_path))
                    cv2.imwrite(save_path, annotated_frame)
                    
                    # Save detailed report
                    txt_name = os.path.splitext(os.path.basename(img_path))[0] + "_zone_report.txt"
                    txt_path = os.path.join(chunk_output_dir, txt_name)
                    
                    with open(txt_path, 'w') as f:
                        f.write("="*60 + "\n")
                        f.write("IOU-BASED ZONE PERSON COUNTING REPORT\n")
                        f.write("="*60 + "\n\n")
                        f.write(f"Frame: {os.path.basename(img_path)}\n")
                        f.write(f"IOU Threshold: {ZONE['iou_threshold']}\n\n")
                        
                        f.write(f"Total People Detected: {total_detected}\n")
                        f.write(f"People in Zone: {count_in_zone}\n")
                        f.write(f"People Outside Zone: {total_detected - count_in_zone}\n\n")
                        
                        f.write("Zone Coordinates (pixels):\n")
                        f.write(f"  Top-Left: ({zone_box[0]}, {zone_box[1]})\n")
                        f.write(f"  Bottom-Right: ({zone_box[2]}, {zone_box[3]})\n\n")
                        
                        if boxes_in_zone:
                            f.write("-"*60 + "\n")
                            f.write("People Inside Zone (with IOU values):\n")
                            f.write("-"*60 + "\n")
                            for i, (box, iou_val) in enumerate(zip(boxes_in_zone, iou_values), 1):
                                f.write(f"\nPerson {i}:\n")
                                f.write(f"  Coordinates: ({int(box[0])}, {int(box[1])}) to ({int(box[2])}, {int(box[3])})\n")
                                f.write(f"  IOU: {iou_val:.4f}\n")
                    
                    # Save YOLO coordinates
                    yolo_txt_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
                    yolo_txt_path = os.path.join(chunk_output_dir, yolo_txt_name)
                    result.save_txt(yolo_txt_path)
                    
                    print(f"[YOLO] ✓ Saved: {save_path}")
                    print(f"[YOLO] Total: {total_detected}, In zone: {count_in_zone}, Outside: {total_detected - count_in_zone}")
                    if iou_values:
                        avg_iou = sum(iou_values) / len(iou_values)
                        print(f"[YOLO] Average IOU: {avg_iou:.4f}")
                    
            except Exception as e:
                print(f"[YOLO ERROR] Failed to process {img_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        yolo_queue.task_done()
        print(f"[YOLO] Completed chunk: {chunk_id}\n")
    
    print("[YOLO] Worker stopped")