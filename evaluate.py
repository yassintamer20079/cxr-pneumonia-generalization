import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score, roc_curve

from data import load_split, transform, XrayDataset    # shared data code (data.py)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# rebuild the same network, then fill it with the trained weights
model = models.resnet18()
model.fc = nn.Linear(512, 1)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model = model.to(device)
model.eval()
print("model loaded")

# run the model on a list of rows -> probabilities and true labels
def predict(rows):
    loader = DataLoader(XrayDataset(rows, transform), batch_size=32, shuffle=False)
    probs = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            z = model(x.to(device)).squeeze(1)
            probs.extend(torch.sigmoid(z).cpu().tolist())
            labels.extend(y.tolist())
    return probs, labels


# check: val AUC should match the 0.9972 from training
val_probs, val_labels = predict(load_split("val"))
print("val AUC:", roc_auc_score(val_labels, val_probs))

# pick the threshold on val: the highest one that still catches >= 95% of pneumonia
fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
i = 0
while tpr[i] < 0.95:
    i += 1
threshold = thresholds[i]
print("threshold:", threshold, "val sensitivity:", tpr[i], "val specificity:", 1 - fpr[i])

# sensitivity and specificity at a fixed threshold
def sens_spec(probs, labels, threshold):
    tp = 0      # pneumonia, flagged      (caught)
    fn = 0      # pneumonia, not flagged  (missed)
    tn = 0      # normal, not flagged     (correctly cleared)
    fp = 0      # normal, flagged         (false alarm)
    for p, y in zip(probs, labels):
        if y == 1.0 and p >= threshold:
            tp += 1
        elif y == 1.0:
            fn += 1
        elif p >= threshold:
            fp += 1
        else:
            tn += 1
    return tp / (tp + fn), tn / (tn + fp)


# Kermany test set: used once, with the threshold frozen from val
test_probs, test_labels = predict(load_split("test"))
test_auc = roc_auc_score(test_labels, test_probs)
test_sens, test_spec = sens_spec(test_probs, test_labels, threshold)
print(f"Kermany test: AUC {test_auc:.4f}, sensitivity {test_sens:.4f}, specificity {test_spec:.4f}")