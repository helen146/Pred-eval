"""
Combine PR curves, histograms, and AUC vs event rate plots for multiple predictors of extreme events.

Reference:
Guth, S. & Sapsis, T.P. (2021). Machine Learning Predictors of Extreme Events Occurring in Complex Dynamical Systems.
Nature Communications, 12, 4868. https://doi.org/10.1038/s41467-021-25113-7

Author: helen146
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Utility functions from your previous code (no plotting here)
def evaluate_at_threshold(a, b, a_thresh):
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]
    y_true = (a > a_thresh).astype(int)
    q = y_true.mean()
    precision, recall, thresholds = precision_recall_curve(y_true, b)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    pr_auc = auc(recall, precision)
    adj_auc = pr_auc - q
    best_idx = np.nanargmax(f1[:-1])
    best_threshold = thresholds[best_idx]
    best_f1        = f1[best_idx]
    best_precision = precision[best_idx]
    best_recall    = recall[best_idx]
    return {
        'a_thresh':      a_thresh,
        'q':             q,
        'precision':     precision,
        'recall':        recall,
        'thresholds':    thresholds,
        'f1':            f1,
        'best_threshold':best_threshold,
        'best_f1':       best_f1,
        'best_precision':best_precision,
        'best_recall':   best_recall,
        'auc':           pr_auc,
        'adjusted_auc':  adj_auc
    }

def compute_prr_surface_volume(a, b, percentiles):
    thresholds = np.percentile(a, percentiles)
    qs, aucs = [], []
    for t in thresholds:
        res = evaluate_at_threshold(a, b, a_thresh=t)
        qs.append(res['q'])
        aucs.append(res['auc'])
    qs = np.array(qs)
    aucs = np.array(aucs)
    order = np.argsort(qs)
    qs_sorted = qs[order]
    aucs_sorted = aucs[order]
    thresholds_sorted = thresholds[order]
    V = np.trapz(aucs_sorted, qs_sorted)
    return V, qs_sorted, aucs_sorted, thresholds_sorted

def find_max_adjusted_auc(a, b, percentiles):
    best_val = -np.inf
    best = None
    for p in percentiles:
        t = np.percentile(a, p)
        res = evaluate_at_threshold(a, b, a_thresh=t)
        adj = res['adjusted_auc']
        if adj > best_val:
            best_val = adj
            best = {**res}
    return best

def align_arrays(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    minlen = min(len(a), len(b))
    return a[:minlen], b[:minlen]

# --- Main plotting script ---

predictor_files = [
    ("Predicted_tau_10 (1).npy", "tau=10"),
    ("Predicted_tau_20.npy", "tau=20"),
    ("Predicted_tau_30 (2).npy", "tau=30"),
]
actual_file = "Actual.npy"
percentiles = np.linspace(80, 99.5, 100)

# Load actual once
Actual = np.load(actual_file).flatten()

# Store results for plotting
pr_curves = []
hist_data = []
auc_curves = []

for pred_file, label in predictor_files:
    predicted = np.load(pred_file).flatten()
    # Align shapes
    a, b = align_arrays(Actual, predicted)
    # PRR surface/AUC curve
    V, qs, aucs, thr = compute_prr_surface_volume(a, b, percentiles)
    # Best threshold for PR/histogram
    best = find_max_adjusted_auc(a, b, percentiles)
    # PR curve at best threshold
    pr = evaluate_at_threshold(a, b, a_thresh=best['a_thresh'])
    pr_curves.append({
        'recall': pr['recall'],
        'precision': pr['precision'],
        'best_recall': pr['best_recall'],
        'best_precision': pr['best_precision'],
        'best_f1': pr['best_f1'],
        'q': pr['q'],
        'label': label,
        'a_thresh': best['a_thresh']
    })
    # Histogram
    hist_data.append({
        'a': a,
        'a_thresh': best['a_thresh'],
        'label': label
    })
    # AUC curve
    auc_curves.append({
        'qs': qs,
        'aucs': aucs,
        'V': V,
        'label': label
    })

# Combined PR curve plot
plt.figure(figsize=(8, 6))
for pr in pr_curves:
    plt.plot(pr['recall'], pr['precision'], label=f"{pr['label']} (F1={pr['best_f1']:.2f})")
    plt.scatter(pr['best_recall'], pr['best_precision'], marker='o')
    plt.hlines(pr['q'], 0, 1, colors='k', linestyles='--', alpha=0.3)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves for All Predictors")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("combined_pr_curves.png")
plt.show()

# Combined histograms
plt.figure(figsize=(8, 6))
for h in hist_data:
    plt.hist(h['a'], bins=50, alpha=0.5, label=h['label'])
    plt.axvline(h['a_thresh'], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual value')
plt.ylabel('Count')
plt.title('Histograms of Actual Values with Extreme Thresholds')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("combined_histograms.png")
plt.show()

# Combined AUC vs event rate plot
plt.figure(figsize=(8, 6))
for ac in auc_curves:
    plt.plot(ac['qs'], ac['aucs'], '-o', label=f"{ac['label']} (V={ac['V']:.3f})")
plt.xlabel('Event rate q')
plt.ylabel('AUC α(q)')
plt.title('AUC vs. Event Rate for All Predictors')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("combined_auc_vs_event_rate.png")
plt.show()