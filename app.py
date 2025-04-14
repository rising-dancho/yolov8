from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Train the model
train_results = model.train(
    data="config.yaml",  # path to dataset YAML
    epochs=1,  # number of training epochs
    imgsz=640,  # image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Perform object detection on an image
results = model("images/sacks.jpg", conf=0.25)
results[0].show()
if results[0].boxes is not None and len(results[0].boxes) > 0:
    print("Bounding boxes detected:", results[0].boxes)
else:
    print("No bounding boxes detected.")

# Run inference
model.predict("images/hollow_blocks.jpg", save=True, imgsz=640, conf=0.2)

# Evaluate model performance on the validation set
metrics = model.val()
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category

# Export the model to tflite format
path = model.export(format="tflite")  # return path to exported model
