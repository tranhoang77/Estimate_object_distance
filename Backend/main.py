import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
from PIL import Image
from UniDepth.unidepth.models import UniDepthV2
from UniDepth.unidepth.utils.camera import Pinhole
from fastapi.middleware.cors import CORSMiddleware
import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model2 = YOLO('yolo11l-seg.pt')

def detect_objects_with_depth(image_np, depth_map, confidence=0.1, iou_threshold=0.2):
    """
    Detect objects using YOLOv8 segmentation and process with depth map information.
    
    Args:
        image_path: Path to the RGB image
        depth_map: Depth map array
        confidence: Detection confidence threshold
        iou_threshold: IoU threshold for determining overlap
        
    Returns:
        List of objects with their distances calculated from segmentation masks
    """
    image = image_np
    
    # Perform object detection with segmentation
    results = model2(image_np)
    class_colors = {}
    boxes = []
    segmentation_overlay = image_np.copy()
    image_w, image_h = image_np.shape[:2]

    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            for i, mask in enumerate(result.masks):
                # Get bounding box
                box = result.boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls)
                class_name = model2.names[class_id]
                conf = float(box.conf)

                if conf < confidence:
                    continue

                # Assign a unique color for each class
                if class_name not in class_colors:
                    class_colors[class_name] = np.random.randint(50, 255, size=(3,), dtype=np.uint8)
                color = class_colors[class_name]

                # Get segmentation mask as numpy array
                mask_array = mask.data.cpu().numpy()[0]  # Shape: [H, W]

                # Resize mask to match image size
                mask_array_resized = cv2.resize(mask_array, (image_h, image_w))  

                # Convert mask to binary (0 or 1)
                mask_binary = (mask_array_resized > 0.5).astype(np.uint8)

                # Apply color mask only to segmented areas
                for c in range(3):
                    segmentation_overlay[:, :, c] = np.where(
                        mask_binary == 1, 
                        (segmentation_overlay[:, :, c] * 0.5 + color[c] * 0.5).astype(np.uint8), 
                        segmentation_overlay[:, :, c]
                    )

                # Store object data
                boxx = {
                    'coords': [x1, y1, x2, y2],
                    'class_name': class_name,
                    'confidence': conf,
                    'mask': mask_array_resized 
                }

                box_distance = calculate_distance_from_mask(depth_map, boxx)
                if box_distance:
                    boxes.append(box_distance)

                    # Draw bbox and label
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, f"{class_name}: {box_distance['distance']:.1f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)


        # Fallback to bounding box method if no masks are available
        else:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get box coordinates
                class_id = int(box.cls)
                class_name = model2.names[class_id]
                conf = float(box.conf)
                
                if conf < confidence:
                    continue
                
                boxx = {
                    'coords': [x1, y1, x2, y2],
                    'class_name': class_name,
                    'confidence': conf
                }
                
                box_distance = calculate_distance(depth_map, boxx)
                
                if box_distance != {}:
                    boxes.append(box_distance)
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, f"{class_name} {conf:.2f} dist: {box_distance['distance']:.2f}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    # Save output image
    cv2.imwrite("assets/output/segmentation_overlay.png", segmentation_overlay)
    cv2.imwrite(f"assets/output/from_web_image.png", image)
    
    # Group boxes by class name
    boxes_by_class = {}
    for box in boxes:
        class_name = box['class_name']
        if class_name not in boxes_by_class:
            boxes_by_class[class_name] = []
        boxes_by_class[class_name].append(box)
    
    # Process overlapping boxes for each class
    final_boxes = []
    for class_name, class_boxes in boxes_by_class.items():
        # Sort boxes by distance
        sorted_boxes = sorted(class_boxes, key=lambda x: x['distance'])
        
        # Keep track of boxes to remove
        boxes_to_remove = set()
        
        # Check for overlaps
        for i in range(len(sorted_boxes)):
            box1 = sorted_boxes[i]
            
            for j in range(i + 1, len(sorted_boxes)):
                box2 = sorted_boxes[j]
                
                # Calculate IoU
                iou = calculate_iou(box1['coords'], box2['coords'])
                
                if iou > iou_threshold:
                    # If boxes overlap significantly, mark the higher distance one for removal
                    boxes_to_remove.add(j)
        
        # Add non-overlapping boxes to final list
        for i, box in enumerate(sorted_boxes):
            if i not in boxes_to_remove:
                final_boxes.append(box)
     
    return final_boxes

def calculate_distance_from_mask(depth_map, box):
    """
    Calculate the average distance of an object using its segmentation mask.
    
    Args:
        depth_map: Depth map array
        box: Dictionary containing object information including mask
        
    Returns:
        Dictionary with object information including average distance
    """
    # Get mask and bounding box coordinates
    mask = box['mask']
    x1, y1, x2, y2 = map(int, box['coords'])
    
    # Ensure coordinates are within image boundaries
    height, width = depth_map.shape
    x1 = max(0, min(x1, width-1))
    y1 = max(0, min(y1, height-1))
    x2 = max(0, min(x2, width-1))
    y2 = max(0, min(y2, height-1))
    
    mask_resized = cv2.resize(mask.astype(np.float32), (x2-x1, y2-y1))
    mask_resized = mask_resized > 0.5  # Convert to binary mask
    
    # Create a full image sized mask
    full_mask = np.zeros((height, width), dtype=bool)
    full_mask[y1:y2, x1:x2] = mask_resized
    
    # Apply mask to depth map
    masked_depth = depth_map.copy()
    masked_depth[~full_mask] = 0
    
    # Calculate average distance for pixels in the mask
    valid_depths = masked_depth[full_mask]
    
    if len(valid_depths) > 0:
        avg_distance = np.median(valid_depths)
        
        object_distance = {
            'class_name': box['class_name'],
            'coords': box['coords'],
            'distance': avg_distance,
            'confidence': box['confidence']
        }
    else:
        # Fallback to center point if mask has no valid depths
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        if center_y < depth_map.shape[0] and center_x < depth_map.shape[1]:
            depth = depth_map[center_y, center_x]
            object_distance = {
                'class_name': box['class_name'],
                'coords': box['coords'],
                'distance': depth,
                'confidence': box['confidence']
            }
        else:
            object_distance = {}
            
    return object_distance

def calculate_distance(depth_map, box):
    x1, y1, x2, y2 = box['coords']

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    if center_y < depth_map.shape[0] and center_x < depth_map.shape[1]:
        depth = depth_map[center_y, center_x]
        object_distance = {
            'class_name' : box['class_name'],
            'coords' : box['coords'],
            'distance' : depth, 
            'confidence' : box['confidence']
        }
    else:
        object_distance = {}
    return object_distance

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: Coordinates [x1, y1, x2, y2] of the first box
        box2: Coordinates [x1, y1, x2, y2] of the second box
        
    Returns:
        IoU value between 0 and 1
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate area of each box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate coordinates of intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if boxes overlap
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    # Calculate area of intersection
    area_intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate area of union
    area_union = area1 + area2 - area_intersection
    
    # Calculate IoU
    iou = area_intersection / area_union
    
    return iou

def load_model():
    name = "unidepth-v2-vits14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device

model, device = load_model()
cocoToVietnamese = {
  "person": "người",
  "bicycle": "xe đạp",
  "car": "xe hơi",
  "motorcycle": "xe máy",
  "airplane": "máy bay",
  "bus": "xe buýt",
  "train": "tàu hỏa",
  "truck": "xe tải",
  "boat": "thuyền",
  "traffic light": "đèn giao thông",
  "fire hydrant": "trụ cứu hỏa",
  "stop sign": "biển báo dừng",
  "parking meter": "đồng hồ tính tiền đỗ xe",
  "bench": "ghế dài",
  "bird": "chim",
  "cat": "mèo",
  "dog": "chó",
  "horse": "ngựa",
  "sheep": "cừu",
  "cow": "bò",
  "elephant": "voi",
  "bear": "gấu",
  "zebra": "ngựa vằn",
  "giraffe": "hươu cao cổ",
  "backpack": "ba lô",
  "umbrella": "ô/dù",
  "handbag": "túi xách",
  "tie": "cà vạt",
  "suitcase": "vali",
  "frisbee": "đĩa bay",
  "skis": "ván trượt tuyết",
  "snowboard": "ván trượt tuyết",
  "sports ball": "bóng thể thao",
  "kite": "diều",
  "baseball bat": "gậy bóng chày",
  "baseball glove": "găng tay bóng chày",
  "skateboard": "ván trượt",
  "surfboard": "ván lướt sóng",
  "tennis racket": "vợt tennis",
  "bottle": "chai",
  "wine glass": "ly rượu vang",
  "cup": "cốc",
  "fork": "nĩa",
  "knife": "dao",
  "spoon": "muỗng",
  "bowl": "bát",
  "banana": "chuối",
  "apple": "táo",
  "sandwich": "bánh sandwich",
  "orange": "cam",
  "broccoli": "bông cải xanh",
  "carrot": "cà rốt",
  "hot dog": "bánh mì kẹp xúc xích",
  "pizza": "bánh pizza",
  "donut": "bánh rán",
  "cake": "bánh kem",
  "chair": "ghế",
  "couch": "ghế sofa",
  "potted plant": "cây trong chậu",
  "bed": "giường",
  "dining table": "bàn ăn",
  "toilet": "nhà vệ sinh",
  "tv": "tivi",
  "laptop": "máy tính xách tay",
  "mouse": "chuột",
  "remote": "điều khiển từ xa",
  "keyboard": "bàn phím",
  "cell phone": "điện thoại di động",
  "microwave": "lò vi sóng",
  "oven": "lò nướng",
  "toaster": "máy nướng bánh mì",
  "sink": "bồn rửa",
  "refrigerator": "tủ lạnh",
  "book": "sách",
  "clock": "đồng hồ",
  "vase": "bình hoa",
  "scissors": "kéo",
  "teddy bear": "gấu bông",
  "hair drier": "máy sấy tóc",
  "toothbrush": "bàn chải đánh răng"
}

@app.post("/map_depth")
async def mapth(file: UploadFile = File(...)):
    try: 
        image_bytes = await file.read()
        
        # Load the RGB image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
        image_np = np.array(image)
        
        print("Image Size (H, W, C):", image_np.shape)

        image_BGR = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        rgb = torch.from_numpy(image_np).permute(2, 0, 1) # C, H, W
        # real_instrinsics = np.load('assets/demo/intrinsics.npy')
        intrinsics = torch.tensor([[[1.1815e+03, 0.0000e+00, 4.8178e+02],
                                    [0.0000e+00, 1.2014e+03, 6.4081e+02],
                                    [0.0000e+00, 0.0000e+00, 1.0000e+00]]], device='cuda:0')
        
        # Giảm tiêu cự khoảng 16-20% để bù sai số
        scale_factor = 0.77  
        intrinsics[0, 0, 0] *= scale_factor  # f_x
        intrinsics[0, 1, 1] *= scale_factor  # f_y
        
        real_intrinsics = intrinsics.to(device)
        camera = Pinhole(K=real_intrinsics)
        predictions = model.infer(rgb, camera)
        
        # Point Cloud in Camera Coordinate
        xyz = predictions["points"]
        xyz_point_cloud = xyz[0].to(torch.float32) 
        
        z_matrix = xyz_point_cloud[2].cpu().numpy()
        
        # Detect objects and calculate distances
        objects_with_distance = detect_objects_with_depth(image_BGR, z_matrix)
        
        # Print results
        string_output = ''
        print("Detected objects with distances:")
        for obj in objects_with_distance:
            print(f"{obj['class_name']}: {obj['distance']:.2f}m (confidence: {obj['confidence']:.2f})")
            string_output += f"có {cocoToVietnamese[obj['class_name']]} ở {obj['distance']:.1f} mét, "
                
        return {"output" : string_output}
    except Exception as e:
        print(f"error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def main():
    return {"message": "Hello World"}