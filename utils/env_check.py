import torch
import sys
import unsloth

print(f"--- System Info ---")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

print(f"\n--- GPU Info ---")
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    print(f"Current CUDA version torch is using: {torch.version.cuda}")

print(f"\n--- Unsloth/Training Info ---")
try:
    import bitsandbytes as bnb
    print(f"Bitsandbytes version: {bnb.__version__}")
    # checks if 4-bit quantization (required for my current GPU for training) will work
    print(f"Bitsandbytes GPU support: Works!")
except Exception as e:
    print(f"Bitsandbytes Error: {e}")

# Check Flash Attention 2 (I need this for Phi 3.5 mini model chosen)
from unsloth import is_bfloat16_supported
print(f"Does GPU support bfloat16? {is_bfloat16_supported()}")