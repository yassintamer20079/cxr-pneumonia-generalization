import csv
from PIL import Image                                # opens image files
from torchvision import transforms                   # ready-made image steps
from torch.utils.data import Dataset, DataLoader     # PyTorch's data tools
from torchvision import models
from torch import nn
import torch
from sklearn.metrics import roc_auc_score

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

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# count pneumonia / normal in the training set
n_pos = 0
for row in train_rows:
    if row["label"] == "PNEUMONIA":
        n_pos += 1
n_neg = len(train_rows) - n_pos

# pneumonia is ~73% of train -> each pneumonia image counts ~1/3, so both classes weigh the same
pos_weight = torch.tensor([n_neg / n_pos], device=device)
print(n_pos, n_neg, pos_weight)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# optimizer: nudges all the weights to reduce the loss (lr = step size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5
best_auc = 0.0

for epoch in range(num_epochs):
    # train: one pass over the 131 training batches
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.float().to(device)
        z = model(x).squeeze(1)
        loss = loss_fn(z, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # evaluate on val: predictions only, no learning
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            z = model(x).squeeze(1)
            all_probs.extend(torch.sigmoid(z).cpu().tolist())
            all_labels.extend(y.tolist())
    auc = roc_auc_score(all_labels, all_probs)

    print(f"epoch {epoch + 1}: train loss {avg_loss:.4f}, val AUC {auc:.4f}")

    # keep the model with the best val AUC
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "best_model.pt")
        print("  saved best model")