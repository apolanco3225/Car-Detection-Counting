import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

from .config import ModelConfig, VehicleConfig, AppConfig
from .model import VehicleDetector

class VehicleDetectionApp:
    def __init__(self):
        """Initialize the Gradio app with model and configurations"""
        self.model_config = ModelConfig()
        self.vehicle_config = VehicleConfig()
        self.app_config = AppConfig()
        
        # Initialize the detector
        self.detector = VehicleDetector(self.model_config, self.vehicle_config)
        
    def visualize_detections(self, image: Image.Image, results: dict, vehicle_counts: dict) -> Image.Image:
        """Visualize the detected vehicles and their counts"""
        plt.figure(figsize=(16, 10))
        plt.imshow(image)
        ax = plt.gca()
        
        # Draw bounding boxes
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            label_name = self.detector.model.config.id2label[label.item()]
            
            if label_name in self.vehicle_config.target_classes:
                color = self.vehicle_config.colors[label_name]
                rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                   fill=False, color=color, linewidth=2)
                ax.add_patch(rect)
                plt.text(box[0], box[1], f'{label_name}: {score:.2f}',
                        color='white', bbox=dict(facecolor=color, alpha=0.5))
        
        # Add count summary
        count_text = f"Vehicle Counts:\n"
        for vehicle, count in vehicle_counts.items():
            count_text += f"{vehicle.capitalize()}: {count}\n"
        
        plt.text(10, 30, count_text, color='white',
                 bbox=dict(facecolor='black', alpha=0.7))
        
        plt.axis('off')
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return Image.open(buf)
    
    def process_image(self, image: Image.Image) -> tuple:
        """Process an image and return the visualization and counts"""
        try:
            # Save image temporarily to process it
            temp_path = "temp_image.jpg"
            image.save(temp_path)
            
            # Detect vehicles
            image, results, vehicle_counts = self.detector.detect_vehicles(temp_path)
            
            # Visualize results
            vis_image = self.visualize_detections(image, results, vehicle_counts)
            
            # Create count text
            count_text = "Vehicle Counts:\n"
            for vehicle, count in vehicle_counts.items():
                count_text += f"{vehicle.capitalize()}: {count}\n"
            
            return vis_image, count_text
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def launch(self):
        """Launch the Gradio interface"""
        interface = gr.Interface(
            fn=self.process_image,
            inputs=gr.Image(type="pil"),
            outputs=[
                gr.Image(type="pil"),
                gr.Textbox(label="Vehicle Counts")
            ],
            title=self.app_config.title,
            description=self.app_config.description,
            examples=self.app_config.examples,
            theme=self.app_config.theme,
            allow_flagging=self.app_config.allow_flagging
        )
        
        return interface.launch()

if __name__ == "__main__":
    app = VehicleDetectionApp()
    app.launch() 