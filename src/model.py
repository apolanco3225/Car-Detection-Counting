import torch
import numpy as np
from PIL import Image
import requests
from transformers import DetrImageProcessor, DetrForObjectDetection
from typing import Tuple, Dict, Any

from .config import ModelConfig, VehicleConfig

class VehicleDetector:
    def __init__(self, model_config: ModelConfig, vehicle_config: VehicleConfig):
        """Initialize the vehicle detector with configurations"""
        self.model_config = model_config
        self.vehicle_config = vehicle_config
        
        # Load model and processor
        self.processor = DetrImageProcessor.from_pretrained(model_config.model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_config.model_name)
        self.model.to(model_config.device)
        
    def load_image(self, image_path: str) -> Image.Image:
        """Load an image from a file path or URL and ensure it's in RGB format"""
        try:
            if image_path.startswith('http'):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            
            # Convert image to RGB if it's not
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
        except Exception as e:
            raise RuntimeError(f"Error loading image: {str(e)}")
    
    def detect_vehicles(self, image_path: str) -> Tuple[Image.Image, Dict[str, Any], Dict[str, int]]:
        """Detect vehicles in an image and return the results"""
        try:
            # Load and process the image
            image = self.load_image(image_path)
            image_np = np.array(image)
            
            # Process the image with DETR processor
            inputs = self.processor(images=image_np, return_tensors="pt")
            inputs = {k: v.to(self.model_config.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Convert outputs to COCO API format
            target_sizes = torch.tensor([image.size[::-1]]).to(self.model_config.device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, 
                threshold=self.model_config.confidence_threshold
            )[0]
            
            # Initialize counters
            vehicle_counts = {vehicle: 0 for vehicle in self.vehicle_config.target_classes}
            
            # Count vehicles
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = self.model.config.id2label[label.item()]
                if label_name in self.vehicle_config.target_classes:
                    vehicle_counts[label_name] += 1
            
            return image, results, vehicle_counts
            
        except Exception as e:
            raise RuntimeError(f"Error during detection: {str(e)}")
    
    def get_detection_info(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detection information from results"""
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model.config.id2label[label.item()]
            if label_name in self.vehicle_config.target_classes:
                detections.append({
                    "label": label_name,
                    "score": score.item(),
                    "box": box.tolist(),
                    "color": self.vehicle_config.colors[label_name]
                })
        return {"detections": detections} 