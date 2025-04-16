from ultralytics import YOLO
import glob

# 1. Load and train the model (saves best.pt inside runs/detect/train*/weights/)
model = YOLO("yolov8n.pt")
model.train(data="config.yaml", epochs=10, device="cpu", imgsz=640)

# 2. Get the latest trained 'best.pt'
best_model_path = sorted(glob.glob("runs/detect/train*/weights/best.pt"))[-1]

# 3. Load the trained YOLOv8 model
trained_model = YOLO(best_model_path)

# 4. (Optional) Evaluate or inference
metrics = trained_model.val()
results = trained_model("images/sacks.jpg", conf=0.25)
results[0].show()

# 5. Export to TorchScript, without touching best.pt
torchscript_path = trained_model.export(format="torchscript")  # creates best.torchscript in same folder

print("Original PyTorch model:", best_model_path)
print("TorchScript model saved to:", torchscript_path)
