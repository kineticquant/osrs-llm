# download_model.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/Phi-3.5-mini-instruct",
    local_dir="models/phi-3.5-mini",
    local_dir_use_symlinks=False,  # copies files fully 
    ignore_patterns=["*.msgpack", "*.h5"]  # skipping any unnecessary large files if present
)

print("Phi-3.5-mini-instruct downloaded successfully to models/phi-3.5-mini!")