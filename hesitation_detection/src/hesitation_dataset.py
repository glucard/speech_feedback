import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

class HesitationDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, resample_rate=16_000, transform=None, target_transform=None):
        self.audio_labels = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.resample_rate = resample_rate
        self.transform = torch.nn.Sequential(
            T.MFCC(sample_rate=resample_rate)
        )
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx]['file_path'])
        waveform, sample_rate = torchaudio.load(audio_path, backend="ffmpeg")

        resampler = T.Resample(sample_rate, self.resample_rate, dtype=waveform.dtype)
        waveform = resampler(waveform)

        label = self.audio_labels.iloc[idx]['votes_for_hesitation']
        if self.transform:
            waveform = self.transform(waveform)
        if self.target_transform:
            label = self.target_transform(label)
        return waveform, label