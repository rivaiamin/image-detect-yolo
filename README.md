# Trash Dump Detection - Web Application

## Pengembangan Awal Model Klasifikasi Sampah Berbasis YOLOv8 dan Faster R-CNN untuk Mitigasi Banjir di Jakarta
- Botol
- Kertas
- Plastik
- Organik
- Logam
- Kaca

A Flask web application for testing the trained YOLOv8 classification model that detects trash dumps in images.

## Features

- 🖼️ **Image Upload**: Support for multiple image formats (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP, HEIC)
- 🎯 **Real-time Prediction**: Instant multi-class classification across kategori sampah (Botol, Kertas, Plastik, Organik, Logam, Kaca)
- 📊 **Confidence Scores**: Display prediction confidence and all class probabilities
- 🎨 **Modern UI**: Beautiful, responsive web interface with drag-and-drop functionality
- 📱 **Mobile Friendly**: Works on desktop and mobile devices

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (if not already done):
   ```bash
   python main.py
   ```

## Usage

1. **Start the Web Application**:
   ```bash
   python web_app.py
   ```

2. **Open Your Browser**:
   - Go to: `http://localhost:5000`
   - Upload an image by clicking the upload area or dragging and dropping
   - View the prediction results

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload and predict image
- `GET /health` - Health check endpoint

## File Structure

```
trash-dump-detection/
├── main.py                 # Training script
├── web_app.py             # Flask web application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface template
├── uploads/              # Temporary upload directory
├── logs/                 # Application logs
├── runs/classify/train/weights/
│   └── best.pt          # Trained model weights
└── dataset/             # Training dataset
```

## Model Information

- **Model Type**: YOLOv8 Classification
- **Classes**:
  - `Botol`
  - `Kertas`
  - `Plastik`
  - `Organik`
  - `Logam`
  - `Kaca`
- **Input Size**: 224x224 pixels
- **Max File Size**: 16MB

## Troubleshooting

1. **Model Not Found**: Make sure you've trained the model first by running `python main.py`
2. **Dependencies Missing**: Install all requirements with `pip install -r requirements.txt`
3. **Port Already in Use**: Change the port in `web_app.py` if port 5000 is occupied

## Example Usage

```python
# Test the model programmatically
from ultralytics import YOLO

model = YOLO('runs/classify/train/weights/best.pt')
results = model('path/to/image.jpg')
print(results)
```

## License

This project is for educational and research purposes.
