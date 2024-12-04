import torch

# Kiểm tra phiên bản PyTorch
print("PyTorch version:", torch.__version__)

# Kiểm tra phiên bản CUDA được PyTorch hỗ trợ
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("cuDNN enabled:", torch.backends.cudnn.enabled)
else:
    print("CUDA is not available.")
