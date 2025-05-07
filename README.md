# Vehicle Detection and Counting

This project uses the DETR (DEtection TRansformer) model to detect and count vehicles (cars, trucks, and motorcycles) in images. It provides a user-friendly web interface built with Gradio.
![Image](https://github.com/user-attachments/assets/0b53c1bf-8e54-40fd-ad22-490f6e1bad75)
## Features

- Vehicle detection using DETR model
- Support for cars, trucks, and motorcycles
- Real-time visualization with bounding boxes
- Vehicle counting statistics
- Web interface using Gradio

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Car-Detection-Counting.git
cd Car-Detection-Counting
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Gradio app:
```bash
python -m src.app
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://127.0.0.1:7860)

3. Upload an image or use the webcam to detect vehicles

## Project Structure

```
src/
├── config.py      # Configuration classes
├── model.py       # DETR model implementation
└── app.py         # Gradio web interface
```

## Configuration

You can modify the following configurations in `src/config.py`:

- Model parameters (confidence threshold, device)
- Target vehicle classes
- Visualization colors
- Gradio interface settings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

