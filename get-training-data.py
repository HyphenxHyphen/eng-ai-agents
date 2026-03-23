from huggingface_hub import snapshot_download

# This downloads the dataset to a local folder named 'drone_data'
snapshot_download(
    repo_id="lgrzybowski/seraphim-drone-detection-dataset", 
    repo_type="dataset", 
    local_dir="./drone_data"
)