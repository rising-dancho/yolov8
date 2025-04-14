from ultralytics import YOLO
model = YOLO("runs/detect/train8/weights/best.pt") # replace with the [actual path] of the best.pt
model.export(format="tflite")