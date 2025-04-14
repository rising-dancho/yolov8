from ultralytics import YOLO
model = YOLO("runs/detect/train8/weights/best.pt")
model.export(format="tflite")