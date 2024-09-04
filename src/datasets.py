import os
import zipfile
import math
import pandas as pd
import torchaudio

from urllib.request import urlretrieve



DATASET_PATH = "temp_datasets"

def hesitation_dev() -> tuple[str, str]:
    """"""
    curr_dataset_path = os.path.join(DATASET_PATH, "hesitation_dev")

    if not os.path.isdir(curr_dataset_path):
        os.mkdir(curr_dataset_path)

    
    # get data dir
    data_path = os.path.join(curr_dataset_path, "dev")

    if not os.path.isdir(data_path):
        zip_file_path = os.path.join(curr_dataset_path, "dev.zip")
        print(f"zip path: {zip_file_path}")
        if not os.path.isfile(zip_file_path):
            print("downloading_dataset...")
            urlretrieve("https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1/resolve/main/dev.zip", zip_file_path)
            print("download done.")

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            print("unzip dataset...")
            zip_ref.extractall(curr_dataset_path)
            print("unzip done.")
    
    annotations_file_path = os.path.join(curr_dataset_path, "annotations.csv")
    
    # get labels file
    if not os.path.isfile(annotations_file_path):
        urlretrieve("https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1/resolve/main/metadata_dev_final.csv", annotations_file_path)
        
        # add duration(sec) column
        annotations = pd.read_csv(annotations_file_path)
        def get_audio_duration(file_path):
            # print(file_path)
            waveform, sample_rate = torchaudio.load(os.path.join(curr_dataset_path, file_path))
            return math.ceil(waveform.shape[-1] / sample_rate)
        annotations['duration(sec)'] = annotations['file_path'].apply(get_audio_duration)
        annotations = annotations.reset_index(drop=True)
        annotations.to_csv(annotations_file_path, index=False)

    return os.path.abspath(annotations_file_path), os.path.abspath(curr_dataset_path)


def hesitation_test() -> tuple[str, str]:
    """"""
    curr_dataset_path = os.path.join(DATASET_PATH, "hesitation_test")

    if not os.path.isdir(curr_dataset_path):
        os.mkdir(curr_dataset_path)
        
    # get data dir
    data_path = os.path.join(curr_dataset_path, "test")

    if not os.path.isdir(data_path):
        zip_file_path = os.path.join(curr_dataset_path, "test.zip")

        print(f"zip path: {zip_file_path}")
        if not os.path.isfile(zip_file_path):
            print("downloading_dataset...")
            urlretrieve("https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1/resolve/main/test.zip", zip_file_path)
            print("download done.")

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            print("unzip dataset...")
            zip_ref.extractall(curr_dataset_path)
            print("unzip done.")
    
    annotations_file_path = os.path.join(curr_dataset_path, "annotations.csv")
    
    # get labels file
    if not os.path.isfile(annotations_file_path):
        urlretrieve("https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1/resolve/main/metadata_test_final.csv", annotations_file_path)
        
        # add duration(sec) column
        annotations = pd.read_csv(annotations_file_path)
        def get_audio_duration(file_path):
            # print(file_path)
            waveform, sample_rate = torchaudio.load(os.path.join(curr_dataset_path, file_path))
            return math.ceil(waveform.shape[-1] / sample_rate)
        annotations['duration(sec)'] = annotations['file_path'].apply(get_audio_duration)
        annotations = annotations.reset_index(drop=True)
        annotations.to_csv(annotations_file_path, index=False)

    return os.path.abspath(annotations_file_path), os.path.abspath(curr_dataset_path)

DATASET_IDS = {
    "hesitation_dev": hesitation_dev,
    "hesitation_test": hesitation_test,
}

def get_data_path(dataset_id:str):
    """
    returns: annotations_file_path, data_dir_path
    """
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    return DATASET_IDS[dataset_id]()

    