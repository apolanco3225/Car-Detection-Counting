import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """Configuration for the DETR model"""
    model_name: str = "facebook/detr-resnet-50"
    confidence_threshold: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class VehicleConfig:
    """Configuration for vehicle detection"""
    target_classes: List[str] = ["car", "truck", "motorcycle"]
    colors: Dict[str, str] = {
        "car": "red",
        "truck": "blue",
        "motorcycle": "green"
    }

@dataclass
class AppConfig:
    """Configuration for the Gradio app"""
    title: str = "Vehicle Detection and Counting"
    description: str = "Detect and count vehicles (cars, trucks, motorcycles) using DETR"
    examples: List[str] = [
        "examples/example1.jpg",
        "examples/example2.jpg"
    ]
    theme: str = "default"
    allow_flagging: bool = False 