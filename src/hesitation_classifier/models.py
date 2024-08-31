import torch
from torch import nn
import torchaudio.transforms as T

class CnnMFCC(nn.Module):
    def __init__(self, sample_rate:int = 16_000, max_frames:int=1_000):
        super(CnnMFCC, self).__init__()

        self.sample_rate = sample_rate
        self.max_frames = max_frames

        self.feature_extractor = nn.Sequential(
            T.MFCC(sample_rate=self.sample_rate)
        )
        
        self.net = nn.Sequential(
            nn.Conv2d(5,5)
        )