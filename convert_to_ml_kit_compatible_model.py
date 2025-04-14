from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers.object_detector import MetadataWriter
from tflite_support.metadata_writers.writer_utils import load_file

# Paths
MODEL_PATH = "best.tflite"
LABEL_FILE = "labels.txt"
EXPORT_DIR = "exported_model"

# Write the label file
with open(LABEL_FILE, "w") as f:
    f.write("\n".join([
        "Bistay_sand",
        "Cement",
        "Gravel",
        "Hollow_blocks",
        "Rebar",
        "Sack",
        "Skim_coat"
    ]))

# Create the metadata writer
writer = MetadataWriter.create_for_inference(
    model_buffer=load_file(MODEL_PATH),
    input_norm_mean=[0.0],         # ML Kit expects 0-1 range normalization
    input_norm_std=[255.0],
    label_file_path=LABEL_FILE,
    input_image_width=640,
    input_image_height=640,
    input_type=object_detector.InputType.FLOAT
)

# Populate metadata and save
writer_utils = writer.populate()
tflite_with_metadata_path = f"{EXPORT_DIR}/best_with_metadata.tflite"
writer_utils.save(tflite_with_metadata_path)

print(f"✅ Metadata added. Saved to: {tflite_with_metadata_path}")
