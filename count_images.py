from pathlib import Path

# count how many X-rays are in each folder

data_dir = Path("data/kermany")

for split in ["train", "val", "test"]:
    for label in ["NORMAL", "PNEUMONIA"]:
        folder = data_dir / split / label
        count = len(list(folder.glob("*.jpeg")))
        print(split, label, count)