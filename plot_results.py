import csv
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

# read the 18 runs -> test AUCs grouped by (starting point, fraction)
aucs = {}
with open("exp2_results.csv", newline="") as f:
    for row in csv.DictReader(f):
        key = (row["start"], float(row["fraction"]))
        if key not in aucs:
            aucs[key] = []
        aucs[key].append(float(row["test_auc"]))

fractions = [0.01, 0.1, 1.0]
percents = [1, 10, 100]

plt.figure(figsize=(7, 5))

# one line per starting point: mean over the 3 seeds, error bar = spread between seeds
for start, name in [("kermany", "Start: Kermany model"), ("imagenet", "Start: ImageNet")]:
    means = []
    stds = []
    for fr in fractions:
        means.append(statistics.mean(aucs[(start, fr)]))
        stds.append(statistics.stdev(aucs[(start, fr)]))
    plt.errorbar(percents, means, yerr=stds, marker="o", capsize=4, label=name)

# reference lines from evaluate.py (Exp 1)
plt.axhline(0.9978, color="gray", linestyle=":", label="Kermany test (same hospital)")
plt.axhline(0.9337, color="red", linestyle="--", label="RSNA, no RSNA labels (Exp 1)")

plt.xscale("log")
plt.ylim(0.925, 1.0)
plt.xticks(percents, ["1%\n(104)", "10%\n(1,040)", "100%\n(10,404)"])
plt.xlabel("RSNA training labels used")
plt.ylabel("RSNA test AUC")
plt.title("Pneumonia model at a new hospital: AUC vs. number of local labels")
plt.legend(loc="lower right", bbox_to_anchor=(1, 0.14))
plt.grid(alpha=0.3)
plt.tight_layout()

Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/auc_vs_labels.png", dpi=150)
print("saved figures/auc_vs_labels.png")