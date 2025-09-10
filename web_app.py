#!/usr/bin/env python3
"""
Trash Dump Detection - Web Application
A Flask web application for testing the trained YOLOv8 classification model
"""

import os
import sys
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
import traceback

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Register HEIF opener for HEIC/HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    logger.info("HEIF/HEIC support enabled")
except ImportError:
    logger.warning("pillow-heif not installed. HEIC/HEIF files may not be supported")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'trash-dump-detection-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Model configuration
MODEL_PATH = 'runs/classify/train/weights/best.pt'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'heic'}

# Global model variable
model = None

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained YOLOv8 model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image_path):
    """Preprocess the image for prediction"""
    try:
        # Try to read with OpenCV first
        image = cv2.imread(image_path)
        
        if image is None:
            # If OpenCV fails, try with PIL (handles HEIC/HEIF and other formats)
            try:
                pil_image = Image.open(image_path)
                # Convert PIL image to numpy array
                if pil_image.mode == 'RGBA':
                    # Convert RGBA to RGB
                    pil_image = pil_image.convert('RGB')
                elif pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Convert PIL RGB to OpenCV BGR
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                logger.info(f"Successfully loaded image using PIL: {image_path}")
            except Exception as pil_error:
                logger.error(f"Failed to load image with PIL: {str(pil_error)}")
                return None
        
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (224x224 for YOLOv8 classification)
        image_resized = cv2.resize(image_rgb, (224, 224))
        
        return image_resized
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(image_path):
    """Make prediction on the uploaded image"""
    try:
        if model is None:
            return None, None, "Model not loaded"
        
        # Run prediction
        results = model(image_path)
        
        # Extract prediction results
        if results and len(results) > 0:
            result = results[0]
            
            # Get class names and probabilities
            if hasattr(result, 'probs') and result.probs is not None:
                probs = result.probs.data.cpu().numpy()
                class_names = model.names
                
                # Get top prediction
                top_class_idx = probs.argmax()
                confidence = float(probs[top_class_idx])
                predicted_class = class_names[top_class_idx]
                
                # Get all class probabilities
                all_predictions = []
                for i, (class_id, class_name) in enumerate(class_names.items()):
                    all_predictions.append({
                        'class': class_name,
                        'confidence': float(probs[i])
                    })
                
                # Sort by confidence
                all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Convert image to base64 for display
                image_base64 = None
                try:
                    with open(image_path, 'rb') as img_file:
                        image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                except Exception as e:
                    logger.warning(f"Could not encode image: {str(e)}")
                
                return {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'all_predictions': all_predictions
                }, image_base64, None
            else:
                return None, None, "No prediction results found"
        else:
            return None, None, "No results returned from model"
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None, None, str(e)

def convert_to_jpg(filepath):
    """Convert file to jpg format"""
    try:
        # Open the image
        img = Image.open(filepath)
        
        # Convert to RGB if necessary (for RGBA, P mode, etc.)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Create new filename with .jpg extension
        base_name = os.path.splitext(filepath)[0]
        jpg_path = f"{base_name}.jpg"
        
        # Save as JPEG
        img.save(jpg_path, 'JPEG', quality=95)
        
        # Remove original file if it's different from the new one
        if jpg_path != filepath:
            try:
                os.remove(filepath)
            except:
                pass
        
        logger.info(f"Converted {filepath} to {jpg_path}")
        return jpg_path
        
    except Exception as e:
        logger.error(f"Error converting {filepath} to JPG: {str(e)}")
        return filepath  # Return original path if conversion fails

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):

            # Secure filename and save
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # convert file to jpg
            filepath = convert_to_jpg(filepath)
            
            logger.info(f"File uploaded: {filename}")
            
            # Make prediction
            prediction_result, image_base64, error = predict_image(filepath)
            
            if error:
                # Clean up uploaded file on error
                try:
                    os.remove(filepath)
                except:
                    pass
                return jsonify({'error': f'Prediction failed: {error}'}), 500
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify({
                'success': True,
                'prediction': prediction_result,
                'image': image_base64
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        # show error tracing
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'model_path': MODEL_PATH
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main function to run the web application"""
    print("=" * 60)
    print("Trash Dump Detection - Web Application")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model file not found!")
        print("Please run the training script first:")
        print("  python main.py")
        sys.exit(1)
    
    # Load model
    print("📦 Loading model...")
    if load_model():
        print("✅ Model loaded successfully!")
    else:
        print("❌ Failed to load model!")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
