import numpy as np
import matplotlib.pyplot as plt

# Variants and metrics from the user's table
variants = [
    "kmeans",
    "uniform_random",
    "shuffle_size",
    "length_buckets",
    "pca1_buckets",
    "random_centroid",
    "kmeanspp_one_shot",
]

# Raw metric values in the order:
# Sil, DBI, CHI, Dunn, Coh, Sep, CosIntra, CosInter, Single
data_raw = {
    "kmeans": [-0.01617, 3.6216, 84.33, 0.1819, 1.0803, 0.9183, 0.6300, 0.000757, 0.04833],
    "uniform_random": [-0.21711, 20.24, 0.9977, 0.0502, 1.4002, 0.1252, 0.0888, 0.01579, 0.0],
    "shuffle_size": [-0.21244, 20.41, 0.9919, 0.0366, 1.4022, 0.2336, 0.0846, 0.01454, 0.04833],
    "length_buckets": [-0.21010, 19.42, 1.6395, 0.0491, 1.3966, 0.1544, 0.1091, 0.01758, 0.0],
    "pca1_buckets": [-0.18003, 18.63, 13.66, 0.0456, 1.3299, 0.3869, 0.2869, 0.03759, 0.0],
    "random_centroid": [-0.14412, 5.2554, 27.11, 0.1534, 1.2605, 0.5676, 0.4172, 0.002606, 0.0],
    "kmeanspp_one_shot": [-0.04630, 4.4001, 71.62, 0.1350, 1.1134, 0.8361, 0.5996, 0.001018, 0.0],
}

labels = ["Sil", "DBI", "CHI", "Dunn", "Coh", "Sep", "CosIntra", "CosInter", "Single"]
num_metrics = len(labels)

# Direction: True if higher is better, False if lower is better
higher_better = np.array([
    True,   # Sil ↑
    False,  # DBI ↓
    True,   # CHI ↑
    True,   # Dunn ↑
    False,  # Coh ↓
    True,   # Sep ↑
    True,   # CosIntra ↑
    False,  # CosInter ↓
    False,  # Single ↓
])

# Build a matrix of shape (n_variants, n_metrics)
vals = np.array([data_raw[v] for v in variants])

# Normalize each metric across variants to [0,1], taking direction into account
vals_norm = np.zeros_like(vals, dtype=float)
for j in range(num_metrics):
    col = vals[:, j]
    min_v, max_v = col.min(), col.max()
    denom = max_v - min_v if max_v != min_v else 1.0
    if higher_better[j]:
        vals_norm[:, j] = (col - min_v) / denom
    else:
        vals_norm[:, j] = (max_v - col) / denom

# Radar chart setup
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]  # close the loop

plt.figure(figsize=(9, 9))
ax = plt.subplot(111, polar=True)

for i, v in enumerate(variants):
    data = vals_norm[i].tolist()
    data += data[:1]  # close the loop
    ax.plot(angles, data, marker="o", linewidth=1.5, label=v)
    ax.fill(angles, data, alpha=0.04)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels([])
ax.set_title("All 9 Metrics Radar (Normalized, This Table)", fontsize=14, pad=20)
ax.grid(True, linestyle="--", linewidth=0.5)

ax.legend(bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()
plt.show()
