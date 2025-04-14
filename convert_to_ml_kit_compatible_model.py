import os
from tflite_support.metadata_writers import writer_utils, object_detector

# Use the float32 model here — the one metadata writing supports
MODEL_PATH = os.path.join("convert", "best_float32.tflite")
assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
print("Files in convert:", os.listdir("convert"))

LABEL_FILE = os.path.join("convert", "labels.txt")
EXPORT_DIR = os.path.join("convert", "exported_model")

os.makedirs(EXPORT_DIR, exist_ok=True)

# Write the label file
with open(LABEL_FILE, "w") as f:
    f.write(
        "\n".join(
            [
                "Bistay_sand",
                "Cement",
                "Gravel",
                "Hollow_blocks",
                "Rebar",
                "Sack",
                "Skim_coat",
            ]
        )
    )

# Corrected metadata writer setup
writer = object_detector.MetadataWriter.create_for_inference(
    model_buffer=writer_utils.load_file(MODEL_PATH),
    input_norm_mean=[0.0],
    input_norm_std=[255.0],
    label_file_paths=[LABEL_FILE]  # This is now passed as a list
)

# Populate the metadata and get the result as bytes
metadata_with_model = writer.populate()

# Save the resulting model with metadata
tflite_with_metadata_path = os.path.join(EXPORT_DIR, "best_with_metadata.tflite")
with open(tflite_with_metadata_path, "wb") as f:
    f.write(metadata_with_model)

print(f"✅ Metadata added. Saved to: {tflite_with_metadata_path}")
