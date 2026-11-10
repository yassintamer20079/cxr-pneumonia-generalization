from pathlib import Path

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

# unique patients
patient_ids = set()

for path, label in images:
    patient_ids.add(get_patient_id(path.name))

print(len(patient_ids))
# each patient -> list of their images
patients = {}

for path, label in images:
    pid = get_patient_id(path.name)
    if pid not in patients:
        patients[pid] = []
    patients[pid].append((path, label))
print(len(patients))

