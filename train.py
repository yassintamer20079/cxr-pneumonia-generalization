import csv
from PIL import Image                                # opens image files
from torchvision import transforms                   # ready-made image steps
from torch.utils.data import Dataset, DataLoader     # PyTorch's data tools


# read the split file -> list of rows (dictionaries) for one set
def load_split(split_name):
    rows = []
    with open("kermany_split.csv", newline="") as f:
        reader = csv.DictReader(f)                   # each line -> {'path', 'label', 'split'}
        for row in reader:
            if row["split"] == split_name:
                rows.append(row)
    return rows

train_rows = load_split("train")
val_rows = load_split("val")
test_rows = load_split("test")
print(len(train_rows), len(val_rows), len(test_rows))   # check: 4171 779 906


# recipe applied to every X-ray -> the input format ResNet-18 was trained on
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),     # gray copied into 3 channels (R = G = B)
    transforms.Resize((224, 224)),                   # same size for every image
    transforms.ToTensor(),                           # PIL image -> tensor (3, 224, 224), values 0-1
    transforms.Normalize([0.485, 0.456, 0.406],      # ImageNet mean per channel
                         [0.229, 0.224, 0.225]),     # ImageNet std per channel
])


# list-like object: dataset[i] -> (ready image tensor, label 1.0 / 0.0)
class XrayDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows                             # keep the list of rows
        self.transform = transform                   # keep the recipe

    def __len__(self):
        return len(self.rows)                        # number of images -> len(dataset)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(row["path"])              # open only this image, only when asked
        x = self.transform(image)
        y = 1.0 if row["label"] == "PNEUMONIA" else 0.0   # word -> number for the loss
        return x, y


# one dataset per set, same blueprint and same recipe
train_dataset = XrayDataset(train_rows, transform)
val_dataset = XrayDataset(val_rows, transform)
# test set: kept aside, only used once at the very end

# train: batches of 32, new random order every epoch (the model learns from these)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val: same order every time (only scored, never learned from)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# check: number of batches -> 131 train, 25 val
print(len(train_loader), len(val_loader))