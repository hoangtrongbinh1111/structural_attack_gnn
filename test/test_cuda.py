import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")


import dgl

print("DGL version:", dgl.__version__)
print("Using GPU:", torch.cuda.is_available())
