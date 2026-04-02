import os
from ultralytics import YOLO
from src.config import Config

class FloorPlanDetector:
    def __init__(self, model_path=Config.YOLO_MODEL_PATH):
        """Initializes the YOLOv8 model locally."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}.")
            
        self.model = YOLO(model_path)
        print(f"✅ Loaded YOLO model from {model_path}")

    def detect(self, image_path, confidence=Config.DEFAULT_CONFIDENCE):
        """Runs the image through YOLO and returns the raw result object."""
        # conf controls how strict the model is
        results = self.model(image_path, conf=confidence)
        return results[0]

    def extract_elements(self, result):
        """
        Translates raw YOLO coordinate math into a structured list 
        for GPT-4o and our visualizer, using clean, independent IDs.
        """
        detections = []
        
        # ─── NEW: INDEPENDENT COUNTERS ───
        class_counters = {
            "zone": 0,
            "door": 0,
            "window": 0
        }
        
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = self.model.names[class_id]
            conf = float(box.conf[0].item())
            x, y, w, h = box.xywh[0].tolist()
            
            # Increment the specific counter for this class
            if class_name in class_counters:
                class_counters[class_name] += 1
            else:
                class_counters[class_name] = 1 # Fallback for unexpected classes
                
            # Create a clean ID: ZONE-1, ZONE-2, DOOR-1, WINDOW-1
            unique_id = f"{class_name.upper()}-{class_counters[class_name]}"
            
            detections.append({
                "id": unique_id,
                "class_name": class_name,
                "confidence": round(conf, 2),
                "coordinates": {
                    "x": round(x), 
                    "y": round(y), 
                    "w": round(w), 
                    "h": round(h)
                }
            })
            
        return detections