import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

def evaluate_at_threshold(a, b, a_thresh, plot=False):
    """
    Compute precision–recall metrics for a fixed extreme-event cutoff.
    """
    # binarize
    y_true = (a > a_thresh).astype(int)
    q = y_true.mean()

    # PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, b)

    # F1 per point
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    # area under PR
    pr_auc = auc(recall, precision)
    adj_auc = pr_auc - q

    # pick best F1 (ignore the last trivial point)
    best_idx = np.nanargmax(f1[:-1])
    best_threshold = thresholds[best_idx]
    best_f1        = f1[best_idx]
    best_precision = precision[best_idx]
    best_recall    = recall[best_idx]

    results = {
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

    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='PR Curve')
        plt.scatter(best_recall, best_precision, color='red',
                    label=f"Best F1 = {best_f1:.3f}")
        plt.hlines(q, 0, 1, colors='k', linestyles='--',
                   label=f'Baseline = {q:.3f}')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve at a_thresh = {a_thresh:.3f}")
        plt.legend()
        plt.grid(True)
        plt.show()

    return results

def sweep_extreme_thresholds(a, b, percentiles):
    """
    Find the percentile cutoff that maximizes adjusted AUC = AUC - q.
    """
    best_adj_auc = -np.inf
    best_stats = None
    for p in percentiles:
        a_thresh = np.percentile(a, p)
        res = evaluate_at_threshold(a, b, a_thresh, plot=False)
        if res['adjusted_auc'] > best_adj_auc:
            best_adj_auc = res['adjusted_auc']
            best_stats = {
                'a_thresh':   a_thresh,
                'adjusted_auc': res['adjusted_auc'],
                'precision':  res['precision'],
                'recall':     res['recall'],
                'thresholds': res['thresholds']
            }
    return best_stats

def compute_prr_surface_volume(a, b, percentiles):
    """
    Compute V = ∫ α(q) dq by sweeping `a`-percentiles → event rates q → AUC α(q).
    Returns:
      V, qs_sorted, aucs_sorted, thresholds_sorted
    """
    thresholds = np.percentile(a, percentiles)
    qs, aucs = [], []
    for t in thresholds:
        res = evaluate_at_threshold(a, b, a_thresh=t, plot=False)
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
    """
    Sweep percentiles and pick the threshold t that maximizes α(q) - q.
    Returns a dict with keys:
      'a_thresh', 'q', 'auc', 'adjusted_auc', 'precision', 'recall', 'thresholds'
    """
    best_val = -np.inf
    best = None
    for p in percentiles:
        t = np.percentile(a, p)
        res = evaluate_at_threshold(a, b, a_thresh=t, plot=False)
        adj = res['adjusted_auc']
        if adj > best_val:
            best_val = adj
            best = {
                'a_thresh':      t,
                'q':             res['q'],
                'auc':           res['auc'],
                'adjusted_auc':  adj,
                'precision':     res['precision'],
                'recall':        res['recall'],
                'thresholds':    res['thresholds']
            }
    return best

if __name__ == "__main__":
    Actual    = np.load("Actual.npy")
    predicted = np.load("Predicted_tau_10 (1).npy")

    percentiles = np.linspace(80, 99.5, 100)

    # 1) Volume under PR-Rate surface
    V, qs, aucs, thr = compute_prr_surface_volume(Actual, predicted, percentiles)
    print(f"Volume under PR-Rate surface V = {V:.4f}")

    # 2) Maximum adjusted AUC α*
    best = find_max_adjusted_auc(Actual, predicted, percentiles)
    print(f"Optimal extreme threshold a^ = {best['a_thresh']:.4f}")
    print(f"Event rate q         = {best['q']:.4f}")
    print(f"AUC α(q)             = {best['auc']:.4f}")
    print(f"Max adjusted AUC α*  = {best['adjusted_auc']:.4f}")

    # 3) (Optional) visualize AUC vs q
    plt.plot(qs, aucs, '-o')
    plt.xlabel('Event rate q')
    plt.ylabel('AUC α(q)')
    plt.title('AUC vs. Event Rate')
    plt.grid(True)
    plt.show()

    # 4) (Optional) final PR curve at a^
    eval_best = evaluate_at_threshold(
        Actual, predicted,
        a_thresh=best['a_thresh'],
        plot=True
    )
    print("Best F1:", eval_best['best_f1'])
    print("Precision:", eval_best['best_precision'])
    print("Recall:", eval_best['best_recall'])