import csv
import random
from pathlib import Path

import pydicom
from PIL import Image

rsna_dir = Path("data/rsna")

# patient ID -> class (a patient with several boxes appears several times: keep one)
classes = {}
with open(rsna_dir / "stage_2_detailed_class_info.csv", newline="") as f:
    for row in csv.DictReader(f):
        classes[row["patientId"]] = row["class"]

# keep Normal (-> NORMAL) and Lung Opacity (-> PNEUMONIA), leave out the third class
patients = []
for patient_id, c in classes.items():
    if c == "Normal":
        patients.append((patient_id, "NORMAL"))
    elif c == "Lung Opacity":
        patients.append((patient_id, "PNEUMONIA"))

n_pneumonia = 0
for patient_id, label in patients:
    if label == "PNEUMONIA":
        n_pneumonia += 1
print(len(patients), n_pneumonia, n_pneumonia / len(patients))

# shuffle (fixed seed = same split every run), then cut 70 / 15 / 15
patients.sort()
random.seed(42)
random.shuffle(patients)

n = len(patients)
n_train = int(n * 0.7)
n_val = int(n * 0.15)

splits = {
    "train": patients[:n_train],
    "val": patients[n_train : n_train + n_val],
    "test": patients[n_train + n_val:],
}
print(len(splits["train"]), len(splits["val"]), len(splits["test"]))

# convert each DICOM to a 224 x 224 PNG (only once) and save the split table
png_dir = rsna_dir / "png"
png_dir.mkdir(exist_ok=True)

count = 0
with open("rsna_split.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label", "split"])
    for split_name, group in splits.items():
        for patient_id, label in group:
            png_path = png_dir / f"{patient_id}.png"
            if not png_path.exists():
                dicom = pydicom.dcmread(rsna_dir / "stage_2_train_images" / f"{patient_id}.dcm")
                image = Image.fromarray(dicom.pixel_array).resize((224, 224))
                image.save(png_path)
            writer.writerow([png_path, label, split_name])
            count += 1
            if count % 1000 == 0:
                print(count, "done")

print("finished:", count)