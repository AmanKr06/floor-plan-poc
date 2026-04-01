import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Paths
    YOLO_MODEL_PATH = os.path.join("models", "yolov8_custom.pt")
    
    # Inference Settings
    DEFAULT_CONFIDENCE = float(os.getenv("CONFIDENCE_THRESHOLD", 0.25))
    
    # Validation
    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing from the .env file.")