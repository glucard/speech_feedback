import os
from urllib.request import urlretrieve
import zipfile

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

    return annotations_file_path, curr_dataset_path 

DATASET_IDS = {
    "hesitation_dev": hesitation_dev
}

def get_data_path(dataset_id:str):
    """
    returns: annotations_file_path, data_dir_path
    """
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    return DATASET_IDS[dataset_id]()

    