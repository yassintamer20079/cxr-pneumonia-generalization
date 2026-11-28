import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from data import load_split, transform, XrayDataset    # shared data code (data.py)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
threshold = 0.5998      # frozen threshold from evaluate.py (Kermany val, 95% sensitivity)

# the Kermany model (the one from Exp 1)
model = models.resnet18()
model.fc = nn.Linear(512, 1)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model = model.to(device)
model.eval()


# run the model on a list of rows -> probabilities (same as evaluate.py)
def predict(rows):
    loader = DataLoader(XrayDataset(rows, transform), batch_size=32, shuffle=False)
    probs = []
    with torch.no_grad():
        for x, y in loader:
            z = model(x.to(device)).squeeze(1)
            probs.extend(torch.sigmoid(z).cpu().tolist())
    return probs


# Grad-CAM: which regions of the last feature maps pushed the pneumonia score up
def gradcam(row):
    x = transform(Image.open(row["path"])).unsqueeze(0).to(device)   # one image -> batch of 1

    saved = {}
    def save_maps(module, inputs, output):
        output.retain_grad()            # keep the gradient of these maps during backward()
        saved["maps"] = output          # (1, 512, 7, 7): 512 feature maps of 7 x 7

    hook = model.layer4.register_forward_hook(save_maps)    # layer4 = last block before the head
    z = model(x)                        # pneumonia score (before sigmoid)
    hook.remove()
    model.zero_grad()
    z.sum().backward()                  # how much each map value changes the score

    maps = saved["maps"]
    weights = maps.grad.mean(dim=(2, 3), keepdim=True)     # importance of each of the 512 maps
    cam = torch.relu((weights * maps).sum(dim=1))           # weighted sum -> (1, 7, 7), keep positive part
    cam = F.interpolate(cam.unsqueeze(0), size=(224, 224), mode="bilinear")[0, 0]
    cam = cam / (cam.max() + 1e-8)                          # scale to 0-1
    return cam.detach().cpu().numpy()


# pick examples: the most confident images in one group
def pick(rows, probs, label, flagged, n):
    group = []
    for row, p in zip(rows, probs):
        if row["label"] == label and (p >= threshold) == flagged:
            group.append((p, row))
    group.sort(key=lambda pair: pair[0], reverse=flagged)  # flagged: highest first, cleared: lowest first
    return group[:n]


kermany_rows = load_split("test")
rsna_rows = load_split("test", "rsna_split.csv")
kermany_probs = predict(kermany_rows)
rsna_probs = predict(rsna_rows)

examples = [
    ("Kermany: pneumonia, caught", pick(kermany_rows, kermany_probs, "PNEUMONIA", True, 2)),
    ("Kermany: normal, cleared", pick(kermany_rows, kermany_probs, "NORMAL", False, 2)),
    ("RSNA: pneumonia, caught", pick(rsna_rows, rsna_probs, "PNEUMONIA", True, 2)),
    ("RSNA: normal, FALSE ALARM", pick(rsna_rows, rsna_probs, "NORMAL", True, 2)),
]

# grid: one row per group, 2 images each -> X-ray with the heatmap on top
fig, axes = plt.subplots(4, 2, figsize=(6, 12))
for r, (title, group) in enumerate(examples):
    for c in range(2):
        ax = axes[r, c]
        ax.axis("off")
        if c < len(group):
            p, row = group[c]
            image = Image.open(row["path"]).convert("L").resize((224, 224))
            ax.imshow(image, cmap="gray")
            ax.imshow(gradcam(row), cmap="jet", alpha=0.4)
            ax.set_title(f"{title}\np = {p:.2f}", fontsize=9)

plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/gradcam_examples.png", dpi=150)
print("saved figures/gradcam_examples.png")