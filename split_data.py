from pathlib import Path
import random
import csv


# dataset folder
data_dir = Path("data/kermany")

# filename -> patient ID
def get_patient_id(filename):
    if filename.startswith("person"):
        return filename.split("_")[0]
    else:
        return filename.split("-")[-2]

# all images as (path, label) pairs
images = []

for split in ["train", "val", "test"]:
    for label in ["NORMAL", "PNEUMONIA"]:
        folder = data_dir / split / label
        for path in folder.glob("*.jpeg"):
            images.append((path, label))

# each patient -> list of their images
patients = {}

for path, label in images:
    patient_id = get_patient_id(path.name)
    if patient_id not in patients:
        patients[patient_id] = []
    patients[patient_id].append((path, label))
print(len(patients))

# shuffle patients (fixed seed = same split every run)
patient_ids = list(patients.keys())
patient_ids.sort()
random.seed(42)
random.shuffle(patient_ids)

# cut patients into 70% train / 15% val / 15% test
n = len(patient_ids)
n_train = int(n * 0.7)
n_val = int(n * 0.15)

train_ids = patient_ids[:n_train]
val_ids = patient_ids[n_train : n_train + n_val]
test_ids = patient_ids[n_train + n_val:]
print(len(train_ids), len(val_ids), len(test_ids))

# patient list -> list of all their (path, label) pairs
def get_images(id_list):
    result = []
    for patient_id in id_list:
        result.extend(patients[patient_id])
    return result

train_images = get_images(train_ids)
val_images = get_images(val_ids)
test_images = get_images(test_ids)
print(len(train_images), len(val_images), len(test_images))


# check: fare share of pneumonia in each set
def pneumonia_share(image_list):
    count = 0
    for path, label in image_list:
        if label == "PNEUMONIA":
            count += 1
            
    return count / len(image_list)
                
print(pneumonia_share(train_images), pneumonia_share(val_images), pneumonia_share(test_images))

# save the split: one row per image -> path, label, which set
with open("kermany_split.csv" , "w" , newline= "") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label", "split"])
    for path, label in train_images:
        writer.writerow([path, label, "train"])
    for path, label in val_images:
        writer.writerow([path, label, "val"])
    for path, label in test_images:
        writer.writerow([path, label, "test"])
