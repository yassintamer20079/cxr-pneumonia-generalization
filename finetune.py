import csv
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score, roc_curve

from data import load_split, transform, XrayDataset    # shared data code (data.py)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# RSNA rows for each set (same split as Exp 1)
train_rows = load_split("train", "rsna_split.csv")
val_rows = load_split("val", "rsna_split.csv")
test_rows = load_split("test", "rsna_split.csv")
print(len(train_rows), len(val_rows), len(test_rows))   # check: 10404 2229 2230

# val and test never change between runs -> build their loaders once
val_loader = DataLoader(XrayDataset(val_rows, transform), batch_size=32, shuffle=False)
test_loader = DataLoader(XrayDataset(test_rows, transform), batch_size=32, shuffle=False)


# run a model on a loader -> probabilities and true labels
def predict(model, loader):
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            z = model(x.to(device)).squeeze(1)
            probs.extend(torch.sigmoid(z).cpu().tolist())
            labels.extend(y.tolist())
    return probs, labels


# highest threshold that still catches >= 95% of pneumonia (same rule as evaluate.py)
def pick_threshold(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    i = 0
    while tpr[i] < 0.95:
        i += 1
    return thresholds[i]


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


# settings (same for both starting points, so the comparison is fair)
lr = 1e-4
num_epochs = {0.01: 30, 0.1: 10, 1.0: 5}    # fewer images -> more passes


# one run: start from one model, fine-tune on a fraction of RSNA train, score on RSNA test
def run(start, fraction, seed):
    # same seed -> same shuffle -> the 1% is inside the 10%, which is inside the 100%
    shuffled = list(train_rows)
    random.Random(seed).shuffle(shuffled)
    subset = shuffled[:int(len(shuffled) * fraction)]
    torch.manual_seed(seed)                  # same seed -> same training randomness
    train_loader = DataLoader(XrayDataset(subset, transform), batch_size=32, shuffle=True)

    # starting point: the Kermany model, or plain ImageNet ResNet-18
    if start == "kermany":
        model = models.resnet18()
        model.fc = nn.Linear(512, 1)
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(512, 1)
    model = model.to(device)

    # loss: weighted by this subset's own pneumonia share (about 40% on RSNA)
    n_pos = 0
    for row in subset:
        if row["label"] == "PNEUMONIA":
            n_pos += 1
    n_neg = len(subset) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train, score on RSNA val after each epoch, keep the best epoch
    best_auc = 0.0
    for epoch in range(num_epochs[fraction]):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.float().to(device)
            z = model(x).squeeze(1)
            loss = loss_fn(z, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_probs, val_labels = predict(model, val_loader)
        auc = roc_auc_score(val_labels, val_probs)
        print(f"  epoch {epoch + 1}: val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "finetuned.pt")

    # best epoch -> threshold on RSNA val -> score once on RSNA test
    model.load_state_dict(torch.load("finetuned.pt", map_location=device))
    val_probs, val_labels = predict(model, val_loader)
    threshold = pick_threshold(val_probs, val_labels)
    test_probs, test_labels = predict(model, test_loader)
    test_auc = roc_auc_score(test_labels, test_probs)
    test_sens, test_spec = sens_spec(test_probs, test_labels, threshold)
    return test_auc, test_sens, test_spec


# all 18 runs: 2 starting points x 3 fractions x 3 seeds -> one row each in exp2_results.csv
with open("exp2_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["start", "fraction", "seed", "test_auc", "test_sens", "test_spec"])
    for start in ["kermany", "imagenet"]:
        for fraction in [0.01, 0.1, 1.0]:
            for seed in [0, 1, 2]:
                print(start, fraction, seed)
                auc, sens, spec = run(start, fraction, seed)
                print(f"  test AUC {auc:.4f}, sensitivity {sens:.4f}, specificity {spec:.4f}")
                writer.writerow([start, fraction, seed, auc, sens, spec])
                f.flush()           # write the row to disk now, so finished runs survive a crash