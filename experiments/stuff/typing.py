import torch


ImageBatch = torch.Tensor   # shape = (batch_size, num_channels=3, height=32, width=32), dtype = torch.float32.
Logits = torch.Tensor       # shape = (batch_size, num_classes=10), dtype = torch.float32.
TargetBatch = torch.Tensor  # shape = (batch_size,), dtype = torch.long.
