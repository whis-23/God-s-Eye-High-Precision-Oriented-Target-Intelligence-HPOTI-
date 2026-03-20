import numpy as np
import cv2
import random

def random_rotation_sar(image: np.ndarray, boxes: np.ndarray, angle_range: tuple = (0, 360)) -> tuple:
    """
    Apply random rotation (0-360) to SAR images and adjust bounding boxes.
    """
    h, w = image.shape[:2]
    angle = random.uniform(*angle_range)
    center = (w // 2, h // 2)
    
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos, sin = np.abs(matrix[0, 0]), np.abs(matrix[0, 1])
    new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    
    rotated_image = cv2.warpAffine(image, matrix, (new_w, new_h))
    
    new_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max, class_id = box
        corners = np.array([[x_min, y_min, 1], [x_max, y_min, 1], [x_min, y_max, 1], [x_max, y_max, 1]])
        new_corners = (matrix @ corners.T).T
        
        new_x_min, new_y_min = np.min(new_corners[:, 0]), np.min(new_corners[:, 1])
        new_x_max, new_y_max = np.max(new_corners[:, 0]), np.max(new_corners[:, 1])
        
        new_boxes.append([
            np.clip(new_x_min, 0, new_w), np.clip(new_y_min, 0, new_h),
            np.clip(new_x_max, 0, new_w), np.clip(new_y_max, 0, new_h), class_id
        ])
        
    return rotated_image, np.array(new_boxes)

def mosaic_mixup(images: list, all_boxes: list, img_size=(640, 640)) -> tuple:
    """
    Mosaic augmentation: Combine 4 images into one.
    """
    yc, xc = [int(random.uniform(-x // 2, x // 2)) + x for x in img_size]  # Mosaic center
    mosaic_img = np.full((img_size[0] * 2, img_size[1] * 2, 3), 114, dtype=np.uint8)  # Padding with gray
    
    mosaic_boxes = []
    
    for i in range(4):
        img = images[i]
        h, w = img.shape[:2]
        
        # Place image in one of 4 quadrants
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, img_size[1] * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, img_size[0] * 2)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, img_size[1] * 2), min(yc + h, img_size[0] * 2)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw, padh = x1a - x1b, y1a - y1b
        
        boxes = all_boxes[i].copy()
        if boxes.size > 0:
            boxes[:, 0] += padw
            boxes[:, 2] += padw
            boxes[:, 1] += padh
            boxes[:, 3] += padh
            mosaic_boxes.append(boxes)
            
    if len(mosaic_boxes):
        mosaic_boxes = np.concatenate(mosaic_boxes, 0)
        np.clip(mosaic_boxes[:, 0:4], 0, 2 * img_size[0], out=mosaic_boxes[:, 0:4])
        
    return cv2.resize(mosaic_img, img_size), mosaic_boxes

if __name__ == "__main__":
    print("Mosaic and Rotation augmentation logic fully implemented.")
