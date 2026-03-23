from datasets import Dataset, Image
import os

# 1. Create a list of dictionaries with image paths, not bytes yet
data = {
    "image": [f"detections/{img}" for img in os.listdir('detections/') if img.endswith(('.jpg', '.png'))],
    "label": ["drone"] * len(os.listdir('detections/')) # adjust if multiple files exist
}

# 2. Convert to a HF Dataset and "cast" the column to Image type
ds = Dataset.from_dict(data)
ds = ds.cast_column("image", Image())

# 3. Save to Parquet - this handles the bytes correctly!
ds.to_parquet("drone_detections.parquet")