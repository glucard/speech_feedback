import pandas as pd
import os
from random import sample

from sklearn.model_selection import train_test_split

from src.hesitation.utils import balance_has_hesitation, filter_to_csv


def filter0(annotations_file_path:str, train_size:int, test_size_from_val_size:int):
    filter_id = "filter0"
    filtered_annotations = pd.read_csv(annotations_file_path)
    filtered_annotations = filtered_annotations[filtered_annotations['up_votes'] > 1]
    filtered_annotations = filtered_annotations[filtered_annotations['down_votes'] == 0]
    filtered_annotations = filtered_annotations.reset_index(drop=True)
    filtered_annotations['has_hesitation'] = (filtered_annotations['votes_for_hesitation'] > 0).astype(int)
    filtered_annotations = filtered_annotations[filtered_annotations['duration(sec)'] > 3]
    filtered_annotations = filtered_annotations[filtered_annotations['duration(sec)'] < 8]

    # remove repeated has hesitation
    filtered_annotations = balance_has_hesitation(filtered_annotations)

    filtered_annotations = filtered_annotations.reset_index(drop=True)

    train_df, val_df = train_test_split(filtered_annotations, train_size=train_size)
    val_df, test_df = train_test_split(val_df, test_size=test_size_from_val_size)

    train_file_path = filter_to_csv(train_df, annotations_file_path, f"{filter_id}_train")
    val_file_path = filter_to_csv(val_df, annotations_file_path, f"{filter_id}_val")
    test_file_path = filter_to_csv(test_df, annotations_file_path, f"{filter_id}_test")
    
    return train_file_path, val_file_path, test_file_path

def filter1(annotations_file_path:str, train_size:int, test_size_from_val_size:int):
    filter_id = "filter1"
    filtered_annotations = pd.read_csv(annotations_file_path)
    filtered_annotations = filtered_annotations[filtered_annotations['up_votes'] > filtered_annotations['down_votes']]
    filtered_annotations = filtered_annotations.reset_index(drop=True)
    filtered_annotations['has_hesitation'] = (filtered_annotations['votes_for_hesitation'] > 0).astype(int)
    filtered_annotations = filtered_annotations[filtered_annotations['duration(sec)'] > 3]
    filtered_annotations = filtered_annotations[filtered_annotations['duration(sec)'] < 8]

    filtered_annotations = balance_has_hesitation(filtered_annotations)

    filtered_annotations = filtered_annotations.reset_index(drop=True)

    train_df, val_df = train_test_split(filtered_annotations, train_size=train_size)
    val_df, test_df = train_test_split(val_df, test_size=test_size_from_val_size)

    train_file_path = filter_to_csv(train_df, annotations_file_path, f"{filter_id}_train")
    val_file_path = filter_to_csv(val_df, annotations_file_path, f"{filter_id}_val")
    test_file_path = filter_to_csv(test_df, annotations_file_path, f"{filter_id}_test")
    
    return train_file_path, val_file_path, test_file_path


def filter2(annotations_file_path:str, train_size:int, test_size_from_val_size:int):
    filter_id = "filter2"
    filtered_annotations = pd.read_csv(annotations_file_path)
    filtered_annotations = filtered_annotations[filtered_annotations['up_votes'] > 1]
    filtered_annotations = filtered_annotations[filtered_annotations['down_votes'] == 0]
    filtered_annotations = filtered_annotations.reset_index(drop=True)
    filtered_annotations['has_hesitation'] = (filtered_annotations[['votes_for_hesitation','votes_for_filled_pause']] > 0).any(axis=1).astype(int)
    filtered_annotations = filtered_annotations[filtered_annotations['duration(sec)'] > 3]
    filtered_annotations = filtered_annotations[filtered_annotations['duration(sec)'] < 10]

    # remove repeated has hesitation
    filtered_annotations = balance_has_hesitation(filtered_annotations)

    filtered_annotations = filtered_annotations.reset_index(drop=True)

    train_df, val_df = train_test_split(filtered_annotations, train_size=train_size, random_state=1)
    val_df, test_df = train_test_split(val_df, test_size=test_size_from_val_size, random_state=1)

    train_file_path = filter_to_csv(train_df, annotations_file_path, f"{filter_id}_train")
    val_file_path = filter_to_csv(val_df, annotations_file_path, f"{filter_id}_val")
    test_file_path = filter_to_csv(test_df, annotations_file_path, f"{filter_id}_test")
    
    return train_file_path, val_file_path, test_file_path
