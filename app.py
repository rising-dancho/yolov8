from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Train the model
train_results = model.train(
    data="config.yaml",  # path to dataset YAML
    epochs=50,  # number of training epochs
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("images/sample.jpg", conf=0.25)
results[0].show()
if results[0].boxes is not None and len(results[0].boxes) > 0:
    print("Bounding boxes detected:", results[0].boxes)
else:
    print("No bounding boxes detected.")

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
