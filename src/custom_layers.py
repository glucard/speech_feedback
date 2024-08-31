import torch
import torch.nn as nn
import torch.nn.functional as F

class PadOrTruncateLayer(nn.Module):
    def __init__(self, fixed_length):
        super(PadOrTruncateLayer, self).__init__()
        self.fixed_length = fixed_length

    def forward(self, x):
        # Get the original length of the tensor
        original_length = x.size(1)
        
        if original_length > self.fixed_length:
            # Truncate the tensor if it's longer than fixed_length
            x = x[:, :self.fixed_length]
        elif original_length < self.fixed_length:
            # Pad the tensor with zeros if it's shorter than fixed_length
            pad_length = self.fixed_length - original_length
            x = F.pad(x, (pad_length,0 ), "constant", 0)
        
        return x