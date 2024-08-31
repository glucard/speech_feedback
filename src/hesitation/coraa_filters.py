import pandas as pd
import os

from sklearn.model_selection import train_test_split

        
def filter_to_csv(filtered_df:pd.DataFrame, annotations_file_path:str, filter_id:str) -> str:
    filtered_dir = os.path.join(*annotations_file_path.split(os.sep)[:-1], "filtered_annotations")

    if not os.path.isdir(filtered_dir):
        os.mkdir(filtered_dir)

    filtered_annotations_file_path = os.path.join(filtered_dir, f"{filter_id}.csv")
    filtered_df.to_csv(filtered_annotations_file_path, index=False)
    return filtered_annotations_file_path


def filter0(annotations_file_path:str, train_size:int, test_size_from_val_size:int):
    filter_id = "filter0"
    filtered_annotations = pd.read_csv(annotations_file_path)
    filtered_annotations = filtered_annotations[filtered_annotations['up_votes'] > 1]
    filtered_annotations = filtered_annotations[filtered_annotations['down_votes'] == 0]
    filtered_annotations = filtered_annotations.reset_index(drop=True)
    filtered_annotations['has_hesitation'] = (filtered_annotations['votes_for_hesitation'] > 0).astype(int)
    filtered_annotations = filtered_annotations[filtered_annotations['duration(sec)'] > 3]
    filtered_annotations = filtered_annotations[filtered_annotations['duration(sec)'] < 8]
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
    filtered_annotations = filtered_annotations.reset_index(drop=True)

    train_df, val_df = train_test_split(filtered_annotations, train_size=train_size)
    val_df, test_df = train_test_split(val_df, test_size=test_size_from_val_size)

    train_file_path = filter_to_csv(train_df, annotations_file_path, f"{filter_id}_train")
    val_file_path = filter_to_csv(val_df, annotations_file_path, f"{filter_id}_val")
    test_file_path = filter_to_csv(test_df, annotations_file_path, f"{filter_id}_test")
    
    return train_file_path, val_file_path, test_file_path
