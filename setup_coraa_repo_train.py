from huggingface_hub import snapshot_download
from src.datasets import DATASET_PATH
import os
import rarfile

REPO_ID = "gabrielrstan/CORAA-v1.1"


coraa_path = snapshot_download(repo_id=REPO_ID, repo_type="dataset")

curr_dataset_path = os.path.join(DATASET_PATH, "hesitation_train")

if not os.path.isdir(curr_dataset_path):
    os.mkdir(curr_dataset_path)

train_dividido_path = os.path.join(coraa_path, "train_dividido")
# rar_files = [file for file in os.listdir(train_dividido_path) if file.split('.')[-1] == "rar"]


if not os.path.isdir(curr_dataset_path):
    os.mkdir(curr_dataset_path)
rarfile.UNRAR_TOOL = "unrar"
r_f_path = os.path.join(train_dividido_path, 'train.part1.rar')
with rarfile.RarFile(r_f_path, 'r') as rar_ref:
    print(f"unzipping {r_f_path} to {curr_dataset_path}")
    rar_ref.extractall(curr_dataset_path)
    print("unzip done.")