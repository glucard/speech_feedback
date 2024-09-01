import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

import random

class PadOrTruncateLayer(nn.Module):
    def __init__(self, fixed_length:int):
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
    
    
class NoiseLayer(nn.Module):
    def __init__(self, p:int=0.1, noise_factor:int=0.005):
        super(NoiseLayer, self).__init__()
        self.p = p
        self.noise_factor = noise_factor

    def forward(self, waveform):
        if random.random() < self.p:
            noise = torch.randn(waveform.size())
            waveform = waveform + self.noise_factor * noise
        return waveform
    
        
class RandomPitchShiftLayer(nn.Module):
    def __init__(self, p:int=0.05, sample_rate:int=16_000, min_steps:int=-5, max_steps:int=5):
        super(RandomPitchShiftLayer, self).__init__()
        self.p = p
        self.sample_rate = sample_rate
        self.min_steps = min_steps
        self.max_steps = max_steps

    def forward(self, waveform):
        if random.random() < self.p:
            n_steps = random.uniform(self.min_steps, self.max_steps)
            pitch_shift = T.PitchShift(self.sample_rate, n_steps)
            waveform = pitch_shift(waveform)
        return waveform    
        
class ReverberationLayer(nn.Module):
    def __init__(self, min_reverb:int=1, max_reverb:int=400):
        super(ReverberationLayer, self).__init__()
        self.min_reverb = min_reverb
        self.max_reverb = max_reverb

    def forward(self, waveform):
        impulse_response = torch.randn(self.min_reverb, self.max_reverb)  # Example impulse response
        return torch.nn.functional.conv1d(waveform.unsqueeze(0), impulse_response.unsqueeze(0)).squeeze(0)