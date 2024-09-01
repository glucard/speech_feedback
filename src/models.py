import torch
from torch import nn
import torchaudio.transforms as T

from src.custom_layers import PadOrTruncateLayer

class CnnMFCC(nn.Module):
    def __init__(self, n_classes:int, sample_rate:int = 16_000, max_audio_length_seconds:int=5):
        super(CnnMFCC, self).__init__()

        self.sample_rate = sample_rate
        self.max_audio_length_seconds = max_audio_length_seconds
        self.features_in = max_audio_length_seconds * sample_rate

        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 256
        n_mfcc = 256

        self.feature_extractor = nn.Sequential(
            # T.MelSpectrogram(
            #     sample_rate=sample_rate,
                # n_fft=n_fft,
                # win_length=win_length,
                # hop_length=hop_length,
                # center=True,
                # pad_mode="reflect",
                # power=2.0,
                # norm="slaney",
                # n_mels=n_mels,
                # mel_scale="htk",
            # )
            T.MFCC(
                sample_rate=self.sample_rate,
                # n_mfcc=n_mfcc,
                # melkwargs={
                #     "n_fft": n_fft,
                #     "n_mels": n_mels,
                #     "hop_length": hop_length,
                #     "mel_scale": "htk",
                #     },
                ),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, (8,8), (2,2),padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.7, inplace=True),

            nn.Conv2d(16, 32, (5,5), (2,2),padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.7, inplace=True),
            
            nn.Conv2d(32, 32, (3,3), (1,1),padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.7, inplace=True),
            
            nn.Flatten(start_dim=1, end_dim=-1)
        )
        linear_in = self.conv(self.feature_extractor(torch.rand(1, 1, sample_rate*max_audio_length_seconds))).shape[-1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.7, inplace=True),
            nn.Linear(in_features=linear_in, out_features=n_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.conv(x)
        x = self.classifier(x)
        return x