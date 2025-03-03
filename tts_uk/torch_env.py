import torch

seed = 1234

# use_mps = torch.mps.is_available()
use_mps = False
use_cuda = torch.cuda.is_available()

if use_mps:
    device = "mps"
    torch.mps.manual_seed(seed)
elif use_cuda:
    device = "cuda"
    torch.cuda.manual_seed(seed)
else:
    device = "cpu"
    torch.manual_seed(seed)

print(f"Inference device: {device}")
