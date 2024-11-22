import os
import zipfile
import math
import pandas as pd
import torchaudio
import re

from urllib.request import urlretrieve
from huggingface_hub import snapshot_download
from os.path import join

from .utils import cut_audio_and_save, fill_not_hesitation


DATASET_PATH = "temp_datasets"

def alip_sample() -> tuple[str, str]:
    """alip
    """
    curr_dataset_path = os.path.join(DATASET_PATH, "alip_sample")

    if not os.path.isdir(curr_dataset_path):
        raise FileNotFoundError(f"Alip Sample Dataset not found. {curr_dataset_path} not exist.")
        
    # get data dir
    data_path = os.path.join(curr_dataset_path, "audios")

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Alip Sample Data not found. {data_path} not exist.")
    
    annotations_file_path = os.path.join(curr_dataset_path, "annotations.csv")
    
    # get labels file
    if not os.path.isfile(annotations_file_path):
        def fix_audio_path(file_path):
            file_path = re.sub(r"^[^-]*-", "", file_path)
            file_path = join("audios", file_path)
            return file_path
        # add duration(sec) column
        annotations = pd.read_csv("temp_datasets/alip_sample/SmallSample-AC26.csv")

        annotations['file_path'] = annotations['File_Name'].map(fix_audio_path)

        annotations = fill_not_hesitation(annotations)

        for audio_path, audio_group in annotations.groupby("file_path"):
            save_dir_path = "cutted_audios"
            annotations.loc[audio_group.index, "file_path"] = audio_group.apply(
                lambda row: cut_audio_and_save(
                    dataset_path=curr_dataset_path,
                    audio_path=audio_path,
                    start=row["Start"],
                    end=row["End"],
                    save_dir_path=save_dir_path), axis=1)

        annotations = annotations.reset_index(drop=True)
        annotations["has_hesitation"] = annotations["Has_Hesitation"].map(lambda x: 1 if x=="Y" else 0)
        annotations.drop(columns=["File_Name", "Start", "End", "Original_Length", "Has_Hesitation"], inplace=True)
        annotations.to_csv(annotations_file_path, index=False)

    return os.path.abspath(annotations_file_path), os.path.abspath(curr_dataset_path)

DATASET_IDS = {
    "alip_sample": alip_sample,
}

def get_data_path(dataset_id:str):
    """
    returns: annotations_file_path, data_dir_path
    """
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    return DATASET_IDS[dataset_id]()

    