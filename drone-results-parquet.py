import pandas as pd
import os

data = []
img_dir = 'assignments/assignment-3/detections' 

for img_name in os.listdir(img_dir):
    if img_name.endswith(('.jpg', '.png')):
        with open(os.path.join(img_dir, img_name), 'rb') as f:
            image_bytes = f.read()
        
        data.append({
            "image": {"bytes": image_bytes, "path": img_name},
            "label": "drone",
            "video_source": img_name.split('_')[0]
        })

df = pd.DataFrame(data)
df.to_parquet("drone_detections.parquet", engine='pyarrow')

print("Done")