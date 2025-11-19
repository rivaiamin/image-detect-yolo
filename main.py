# install YOLOv8 if not already
# pip install ultralytics

import os
import argparse
from pathlib import Path

# Set datasets_dir to the current project's dataset directory
# This allows each project to have its own dataset path without modifying global settings
project_root = Path(__file__).parent.absolute()
datasets_dir = project_root / "dataset"

# Update Ultralytics settings before importing YOLO
from ultralytics.utils import SETTINGS
SETTINGS.update({'datasets_dir': str(datasets_dir)})

from ultralytics import YOLO

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 model for trash dump detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start fresh training
  python main.py --epochs 50
  
  # Resume from last checkpoint
  python main.py --resume runs/detect/train/weights/last.pt
  
  # Resume from best checkpoint
  python main.py --resume runs/detect/train/weights/best.pt
  
  # Resume from custom checkpoint path
  python main.py --resume /path/to/checkpoint.pt --epochs 100
        """
    )
    parser.add_argument(
        '--resume', '--resume-from',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from (e.g., runs/detect/train/weights/last.pt)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training (default: 640)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size for training (default: 16)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8s.pt',
        help='Base model to use (default: yolov8s.pt). Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load model or checkpoint
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"❌ Error: Checkpoint file not found: {resume_path}")
            print(f"   Looking for checkpoint at: {resume_path.absolute()}")
            return
        print(f"📂 Resuming training from checkpoint: {resume_path}")
        model = YOLO(str(resume_path))
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"⚠️  Warning: Model file not found: {model_path}")
            print(f"   Using default model: {args.model}")
        print(f"🆕 Starting fresh training with model: {args.model}")
        model = YOLO(args.model)
    
    # 2. Train on your dataset
    # Dataset structure should be:
    # dataset/
    #   data.yaml          # dataset configuration file
    #   train/
    #       images/
    #           img1.jpg
    #           img2.jpg
    #       labels/
    #           img1.txt
    #           img2.txt
    #   valid/
    #       images/
    #       labels/
    #   test/
    #       images/
    #       labels/
    
    print(f"🚀 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Batch size: {args.batch}")
    
    train_kwargs = {
        'data': str(project_root / "dataset" / "data.yaml"),  # absolute path to your dataset YAML config file
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch
    }
    
    # If resuming, pass resume=True to continue from the checkpoint
    if args.resume:
        train_kwargs['resume'] = True
    
    model.train(**train_kwargs)
    
    # 3. Validate the trained model
    print("📊 Validating model...")
    model.val()
    
    # 4. Run prediction on a new image (if test image exists)
    test_image = project_root / "test.jpg"
    if test_image.exists():
        print(f"🔍 Running prediction on {test_image}...")
        results = model(str(test_image))
        print(results)
    else:
        print(f"ℹ️  Test image not found: {test_image}")

if __name__ == '__main__':
    main()
