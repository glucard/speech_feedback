import pandas as pd
import random
import os

def balance_has_hesitation(df: pd.DataFrame) -> pd.DataFrame:
    # remove repeated has hesitation
    has_hesitation_count = df['has_hesitation'].value_counts()[1]
    not_has_hesitation_count = df['has_hesitation'].value_counts()[0]
    to_remove = random.sample(list(df[df['has_hesitation'] == 0].index), not_has_hesitation_count-has_hesitation_count)
    return df.drop(to_remove)

def filter_to_csv(filtered_df:pd.DataFrame, annotations_file_path:str, filter_id:str) -> str:
    filtered_dir = os.path.join(*annotations_file_path.split(os.sep)[:-1], "filtered_annotations")

    if not os.path.isdir(filtered_dir):
        os.mkdir(filtered_dir)

    filtered_annotations_file_path = os.path.join(filtered_dir, f"{filter_id}.csv")
    filtered_df.to_csv(filtered_annotations_file_path, index=False)
    return filtered_annotations_file_path