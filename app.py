from ultralytics import YOLO
import glob

# Load a model and train it
model = YOLO("yolov8n.pt")
model.train(
    data="config.yaml",
    epochs=50,
    device="cpu",
)

# Automatically find the latest trained model
best_model_path = sorted(glob.glob("runs/detect/train*/weights/best.pt"))[-1]
trained_model = YOLO(best_model_path)

# Evaluate the trained model (optional)
metrics = trained_model.val()

# Run inference (optional)
results = trained_model("images/sample.jpg", conf=0.25)
results[0].show()

# ✅ Export trained modelo
path = trained_model.export(format="tflite", half=True) # Works directly with Google ML Kit's ObjectDetector
print("Model exported to:", path)