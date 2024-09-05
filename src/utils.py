import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import librosa
import torch
import os

from src.custom_audio_dataset import CustomAudioDataset

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


# Custom collate function to pad audio samples in a batch
def pad_collate_fn(batch):
    # Extract the waveforms and targets from the batch
    batch_size = len(batch)
    data = [item[0][0].reshape(-1, 40) for item in batch]
    
    targets = [item[1] for item in batch]

    # Pad the waveforms to the length of the longest one
    
    data = pad_sequence(data, batch_first=True)

    # Stack the targets into a tensor
    targets = torch.tensor(targets)

    return data.reshape(batch_size, 40, -1), targets

def dataloader_from_filter(
        annotations_file_path:str,
        data_dir_path:str,
        filter,
        target_column_name:str,
        train_data_transform:torch.nn.Sequential=None,
        data_transform:torch.nn.Sequential=None,
        train_size:float=0.6,
        test_from_val_size:float=0.5) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    create a dataset from a filter then splits it on train, val and test.
    
    ### args
		__filter__ function from coraa_filters.
    ### returns
		#### str, str, str
		__train_filter_annotations_file__ train csv path from filter.
		__val_filter_annotations_file__:str - val csv path from filter.
		__test_filter_annotations_file__:str - test csv path from filter.
    """
    assert os.path.isfile(annotations_file_path), f"{annotations_file_path} not found."
    assert os.path.isdir(data_dir_path), f"{data_dir_path} not found."
    
    train_filter_annotations_file, val_filter_annotations_file, test_filter_annotations_file = filter(
		    annotations_file_path, train_size=train_size, test_size_from_val_size=test_from_val_size
		)
    
    train_filter_dataset = CustomAudioDataset(train_filter_annotations_file, data_dir_path, column_predict_name=target_column_name, data_transform=train_data_transform)
    val_filter_dataset = CustomAudioDataset(val_filter_annotations_file, data_dir_path, column_predict_name=target_column_name, data_transform=data_transform)
    test_filter_dataset = CustomAudioDataset(test_filter_annotations_file, data_dir_path, column_predict_name=target_column_name, data_transform=data_transform)

    # Usage in DataLoader
    train_filter_dataloader = torch.utils.data.DataLoader(
        dataset=train_filter_dataset,
        batch_size=256,
        # collate_fn=pad_collate_fn
    )
    val_filter_dataloader = torch.utils.data.DataLoader(
        dataset=val_filter_dataset,
        batch_size=256,
        # collate_fn=pad_collate_fn
    )
    test_filter_dataloader = torch.utils.data.DataLoader(
        dataset=test_filter_dataset,
        batch_size=256,
        # collate_fn=pad_collate_fn
    )

    return train_filter_dataloader, val_filter_dataloader, test_filter_dataloader