import csv
from pathlib import Path

import matplotlib.pyplot as plt

CSV_PATH = Path("donné_teste_rsult/all_vs_all_20260623_131657.csv")
OUT_DIR = Path("images/text_similarity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))

for row in rows:
    for key in ("TFIDF", "SBERT", "HYBRID"):
        row[key] = float(row[key])

plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.boxplot(
    [[row[key] for row in rows] for key in ("TFIDF", "SBERT", "HYBRID")],
    tick_labels=["TF-IDF", "SBERT", "Hybride"],
    patch_artist=True,
    boxprops={"facecolor": "#dceaf7", "edgecolor": "#24527a"},
    medianprops={"color": "#c44e52", "linewidth": 1.5},
)
ax.set_ylabel("Similarite")
ax.set_ylim(0, 1)
ax.set_title("Distribution des scores de similarite textuelle")
fig.tight_layout()
fig.savefig(OUT_DIR / "distributions_scores.png", dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.scatter(
    [row["TFIDF"] for row in rows],
    [row["SBERT"] for row in rows],
    c=[row["HYBRID"] for row in rows],
    cmap="viridis",
    alpha=0.72,
    edgecolors="white",
    linewidths=0.35,
)
ax.set_xlabel("Score TF-IDF")
ax.set_ylabel("Score SBERT")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Relation entre TF-IDF et SBERT")
colorbar = fig.colorbar(ax.collections[0], ax=ax)
colorbar.set_label("Score hybride")
fig.tight_layout()
fig.savefig(OUT_DIR / "tfidf_sbert_scatter.png", dpi=220)
plt.close(fig)

ranking = sorted(rows, key=lambda row: row["HYBRID"], reverse=True)[:10]
labels = [f"{row['E1']}-{row['E2']}" for row in reversed(ranking)]
values = [row["HYBRID"] for row in reversed(ranking)]
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.barh(labels, values, color="#4c956c")
ax.set_xlabel("Score hybride")
ax.set_xlim(0, 1)
ax.set_title("Dix paires les plus proches selon le score hybride")
for index, value in enumerate(values):
    ax.text(value + 0.015, index, f"{value:.3f}", va="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "top_paires_hybride.png", dpi=220)
plt.close(fig)
