import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import librosa
import torch

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