import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score

from data import load_split, transform, XrayDataset    # shared data code (data.py)


train_rows = load_split("train")
val_rows = load_split("val")
test_rows = load_split("test")
print(len(train_rows), len(val_rows), len(test_rows))   # check: 4171 779 906

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

# pretrained ResNet-18, last layer replaced: 512 features -> 1 score (pneumonia)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 1)

# Mac GPU if available, otherwise CPU
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

# weighted binary cross-entropy (sigmoid included)
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