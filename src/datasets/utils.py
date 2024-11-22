import torchaudio
import os
import math
import random

import pandas as pd

from sklearn.model_selection import train_test_split
from os.path import join

def cut_audio_and_save(dataset_path:str, audio_path:str, start:float, end:float, save_dir_path:str):
    """
    Args:
        audio_path: audio path.
        start: cut start in seconds.
        end: cut end in seconds.
    """
    wav, sr = torchaudio.load(join(dataset_path, audio_path))
    filename_path = join(save_dir_path, f"{start}_{os.path.splitext(os.path.basename(audio_path))[0]}.wav")
    
    start = int(sr * start)
    end = int(sr * end)
    wav = wav[:, start:end]

    dir_path = join(dataset_path, save_dir_path)
    full_filename_path = join(dataset_path, filename_path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    torchaudio.save(full_filename_path, wav, sr)
    return filename_path

def get_audio_duration(file_path:str):
    if not os.path.isfile(file_path):
        raise FileExistsError(f"Cant find {file_path}")
    
    waveform, sample_rate = torchaudio.load(file_path)
    return math.ceil(waveform.shape[-1] / sample_rate)

def find_not_intersect_start_spot(annotations:pd.DataFrame, duration:int):
    start_min = annotations["Start"].min()
    if duration <= start_min:
        free_space = start_min - duration
        return free_space * random.random()
    
    annotations = annotations.sort_values(by="Start")
    free_lines = list(zip(annotations["Start"].tolist()[1:], annotations["End"].tolist()[:-1]))
    random.shuffle(free_lines)
    for x1, x0 in free_lines:
        line_width = x1 - x0
        if duration <= line_width:
            free_space = line_width - duration
            return x0 + free_space * random.random()

    return None

def fill_not_hesitation(annotations:pd.DataFrame):
    new_annotations = []

    annotations["duration"] = annotations["End"] - annotations["Start"]
    for audio_path, audio_group in annotations.groupby("file_path"):
        # TODO melhorar distribuição
        durations = audio_group["duration"].tolist()
        while len(durations) > 0:
            selected_duration = durations.pop()
            selected_start_spot = find_not_intersect_start_spot(audio_group, selected_duration)
            if selected_start_spot:
                new_row = {
                    "ID":audio_group["ID"].iloc[0],
                    "Original_Length":audio_group["Original_Length"].iloc[0],
                    "Start":selected_start_spot,
                    "End":selected_start_spot + selected_duration,
                    "Has_Hesitation":"N",
                    "File_Name":audio_group["File_Name"].iloc[0],
                    "file_path":audio_group["file_path"].iloc[0],
                    "duration":selected_duration,
                }
                audio_group = pd.concat([audio_group, pd.DataFrame([new_row])], ignore_index=True)
        new_annotations.append(audio_group)

    annotations = pd.concat(new_annotations)
    return annotations

def split_train_val_test(annotations_file_path:str, train_size:float, test_size_from_val_size:float) -> str:
    
    annotations = pd.read_csv(annotations_file_path)

    train_df, val_df = train_test_split(annotations, train_size=train_size)
    val_df, test_df = train_test_split(val_df, test_size=test_size_from_val_size)

    split_dir = join(os.path.dirname(annotations_file_path), "split_annotations")
    if not os.path.isdir(split_dir):
        os.mkdir(split_dir)

    train_annotation_path = join(split_dir, "train.csv")
    val_annotation_path = join(split_dir, "val.csv")
    test_annotation_path = join(split_dir, "test.csv")

    train_df.to_csv(train_annotation_path)
    val_df.to_csv(val_annotation_path)
    test_df.to_csv(test_annotation_path)
    
    return train_annotation_path, val_annotation_path, test_annotation_path