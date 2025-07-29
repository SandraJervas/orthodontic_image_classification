import os
import shutil
import random
from sklearn.model_selection import train_test_split
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

INPUT_DIR = "data/processed/images"
OUTPUT_DIR = "data/processed/split"
CLASSES = os.listdir(INPUT_DIR)

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

for cls in CLASSES:
    files = os.listdir(os.path.join(INPUT_DIR, cls))
    files = [f for f in files if f.endswith(('.jpg', '.png'))]

    random.seed(params["split"]["seed"])
    random.shuffle(files)

    trainval, test = train_test_split(files, test_size=params["split"]["test_size"], random_state=params["split"]["seed"])
    train, val = train_test_split(trainval, test_size=params["split"]["val_size"], random_state=params["split"]["seed"])

    for split_name, split_files in zip(["train", "val", "test"], [train, val, test]):
        for file in split_files:
            src = os.path.join(INPUT_DIR, cls, file)
            dst = os.path.join(OUTPUT_DIR, split_name, cls, file)
            shutil.copy2(src, dst)
