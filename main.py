# install YOLOv8 if not already
# pip install ultralytics

from ultralytics import YOLO

# 1. Load a YOLOv8 classification model (pretrained on ImageNet)
model = YOLO("yolov8s-cls.pt")  # n = nano (smallest, fastest). You can also use yolov8s-cls.pt

# 2. Train on your dataset
# Dataset structure should be:
# dataset/
#   train/
#       garbage/
#           img1.jpg
#           img2.jpg
#       clean/
#           img3.jpg
#           img4.jpg
#   val/
#       garbage/
#       clean/

model.train(
    data="dataset",   # path to your dataset root folder
    epochs=20,        # increase if needed
    imgsz=224,        # image size
    batch=16
)

# 3. Validate the trained model
model.val()

# 4. Run prediction on a new image
results = model("test.jpg")  
print(results)        # will show predicted class + confidence
