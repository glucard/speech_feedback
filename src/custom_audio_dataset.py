import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

class CustomAudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, column_predict_name, resample_rate=16_000, data_transform=None, target_transform=None):
        self.audio_labels = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.resample_rate = resample_rate
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.column_predict_name = column_predict_name

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx]['file_path'])
        waveform, sample_rate = torchaudio.load(audio_path, backend="ffmpeg")

        resampler = T.Resample(sample_rate, self.resample_rate, dtype=waveform.dtype)
        waveform = resampler(waveform)

        label = self.audio_labels.iloc[idx][self.column_predict_name]
        if self.data_transform:
            waveform = self.data_transform(waveform)
        if self.target_transform:
            label = self.target_transform(label)
        return waveform, label