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
        # conf controls how strict the model is (0.25 means 25% sure)
        results = self.model(image_path, conf=confidence)
        return results[0]

    def extract_elements(self, result):
        """
        Translates raw YOLO coordinate math into a structured list 
        for GPT-4o and our visualizer.
        """
        detections = []
        
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = self.model.names[class_id]
            conf = float(box.conf[0].item())
            
            # Get coordinates: x_center, y_center, width, height
            x, y, w, h = box.xywh[0].tolist()
            
            detections.append({
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