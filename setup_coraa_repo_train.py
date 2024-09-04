from huggingface_hub import snapshot_download
from src.datasets import DATASET_PATH
import os
import rarfile
import shutil
import pandas as pd
import torchaudio
import math

REPO_ID = "gabrielrstan/CORAA-v1.1"
curr_dataset_path = os.path.join(DATASET_PATH, "hesitation_train")
coraa_path = snapshot_download(repo_id=REPO_ID, repo_type="dataset")

if not os.path.isdir(curr_dataset_path):
    os.mkdir(curr_dataset_path)

    coraa_path = snapshot_download(repo_id=REPO_ID, repo_type="dataset")


    train_dividido_path = os.path.join(coraa_path, "train_dividido")
    # rar_files = [file for file in os.listdir(train_dividido_path) if file.split('.')[-1] == "rar"]

    if not os.path.isdir(os.path.join(curr_dataset_path, "train")):
        print("aqui")
        # rarfile.UNRAR_TOOL = "unrar"
        # r_f_path = os.path.join(train_dividido_path, 'train.part1.rar')
        # with rarfile.RarFile(r_f_path, 'r') as rar_ref:
        #     print(f"unzipping {r_f_path} to {curr_dataset_path}")
        #     rar_ref.extractall(curr_dataset_path)
        #     print("unzip done.")

annotations_file_path = os.path.join(curr_dataset_path, "annotations.csv")
if not os.path.isfile(annotations_file_path):
    from_annotations_path = os.path.join(coraa_path, "metadata_train_final.csv")

    # add duration(sec) column
    annotations = pd.read_csv(from_annotations_path)
    def get_audio_duration(file_path):
        # print(file_path)
        waveform, sample_rate = torchaudio.load(os.path.join(curr_dataset_path, file_path))
        return math.ceil(waveform.shape[-1] / sample_rate)
    annotations['duration(sec)'] = annotations['file_path'].apply(get_audio_duration)
    annotations = annotations.reset_index(drop=True)
    annotations.to_csv(annotations_file_path, index=False)