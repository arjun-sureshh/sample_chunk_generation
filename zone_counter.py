def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes
    
    Formula: IOU = Intersection Area / Union Area
    
    Parameters:
    - box1: [x1, y1, x2, y2] - person's bounding box
    - box2: [x1, y1, x2, y2] - zone box
    
    Returns: IOU value between 0.0 and 1.0
    """
    # Get intersection rectangle coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Check if boxes don't overlap at all
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    # Calculate IOU
    iou = intersection_area / union_area
    
    return iou


def scale_zone_to_image(zone_config, image_width, image_height):
    """
    Convert normalized zone coordinates (0-1) to pixel coordinates
    """
    return [
        int(zone_config['x1'] * image_width),
        int(zone_config['y1'] * image_height),
        int(zone_config['x2'] * image_width),
        int(zone_config['y2'] * image_height)
    ]


def count_people_in_zone(boxes, zone_box, iou_threshold):
    """
    Count people inside zone using IOU threshold
    
    Parameters:
    - boxes: list of person bounding boxes
    - zone_box: zone boundary
    - iou_threshold: minimum IOU value to count as "inside zone"
    
    Returns: (count, boxes_in_zone, iou_values)
    """
    count = 0
    boxes_in_zone = []
    iou_values = []
    
    for box in boxes:
        # Calculate IOU between person box and zone
        iou = calculate_iou(box, zone_box)
        
        # Check if IOU meets threshold
        if iou >= iou_threshold:
            count += 1
            boxes_in_zone.append(box)
            iou_values.append(iou)
    
    return count, boxes_in_zone, iou_values