import torch
from torch import nn
import torchaudio.transforms as T

from src.custom_layers import PadOrTruncateLayer

class CnnMFCC(nn.Module):
    def __init__(self, n_classes:int, sample_rate:int = 16_000, features_in:int=5, dropout:float=0.5):
        super(CnnMFCC, self).__init__()

        self.sample_rate = sample_rate
        self.features_in = features_in

        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 40
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
                melkwargs={
                #     "n_fft": n_fft,
                     "n_mels": n_mels,
                #     "hop_length": hop_length,
                #     "mel_scale": "htk",
                    },
                ),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, (8,8), (2,2),padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout, inplace=True),

            nn.Conv2d(16, 32, (5,5), (2,2),padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout, inplace=True),
            
            nn.Conv2d(32, 32, (3,3), (1,1),padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout, inplace=True),
            
            nn.Flatten(start_dim=1, end_dim=-1)
        )
        linear_in = self.conv(self.feature_extractor(torch.ones(1, 1, features_in))).shape[-1]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(in_features=linear_in, out_features=n_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.conv(x)
        x = self.classifier(x)
        return x
    
class CnnMFCC_v2(nn.Module):
    def __init__(self, n_classes: int, sample_rate: int = 16_000, features_in: int = 5, dropout: float = 0.5):
        super(CnnMFCC_v2, self).__init__()

        self.sample_rate = sample_rate
        self.features_in = features_in

        n_fft = 2048
        hop_length = 512
        n_mels = 40
        n_mfcc = 40  # Assuming MFCCs, typically fewer are used.

        self.feature_extractor = nn.Sequential(
            T.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "n_mels": n_mels,
                    "hop_length": hop_length,
                },
            )
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, (8, 8), stride=(2, 2), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),

            nn.Conv2d(16, 32, (5, 5), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            
            nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            
            nn.Flatten()
        )
        linear_in = self.conv(self.feature_extractor(torch.ones(1, 1, features_in))).shape[-1]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=linear_in, out_features=n_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.conv(x)
        x = self.classifier(x)
        return x
    
class CnnSpectrogram(nn.Module):
    def __init__(self, n_classes: int, sample_rate: int = 16_000, features_in: int = 5, dropout: float = 0.5):
        super(CnnSpectrogram, self).__init__()

        self.sample_rate = sample_rate
        self.features_in = features_in

        n_fft = 2048
        hop_length = 512
        n_mels = 40

        # Feature extractor for spectrogram
        self.feature_extractor = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=None,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0
            )
        )
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            nn.Flatten()
        )
        linear_in = self.conv(self.feature_extractor(torch.ones(1, 1, features_in))).shape[-1]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=linear_in, out_features=n_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.conv(x)
        x = self.classifier(x)
        return x
    
    
    
class CnnSpectrogram_v2(nn.Module):
    def __init__(self, n_classes: int, sample_rate: int = 16_000, features_in: int = 5, dropout: float = 0.5):
        super(CnnSpectrogram_v2, self).__init__()

        self.sample_rate = sample_rate
        self.features_in = features_in

        n_fft = 2048
        hop_length = 512
        n_mels = 40

        # Feature extractor for spectrogram
        self.feature_extractor = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=None,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0
            )
        )
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            nn.Flatten()
        )
        linear_in = self.conv(self.feature_extractor(torch.ones(1, 1, features_in))).shape[-1]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=linear_in, out_features=n_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.conv(x)
        x = self.classifier(x)
        return x