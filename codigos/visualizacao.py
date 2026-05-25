"""
Visualizacao da BoVW projetada em 2D com UMAP e t-SNE.
Saida:
  resultados/visualizacao_bovw_umap.png
  resultados/visualizacao_bovw_tsne.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap
from sklearn.manifold import TSNE


RAIZ = Path("/Users/leticia.neves/Desktop/6sem/2027/leo/entrega3")
FEAT = RAIZ / "features"
SAIDA = RAIZ / "resultados"
SAIDA.mkdir(exist_ok=True)

K_VW = 128


def plot_proj(emb, labels, classnames, titulo, caminho):
    """
    Scatter colorindo apenas as TOP-10 classes (com mais imagens).
    O resto fica cinza claro para nao virar arco-iris ilegivel.
    """
    uniq, counts = np.unique(labels, return_counts=True)
    top10 = uniq[np.argsort(-counts)[:10]]
    top10_set = set(top10.tolist())

    plt.figure(figsize=(11, 9))
    # primeiro o "resto"
    mask_resto = np.array([l not in top10_set for l in labels])
    plt.scatter(emb[mask_resto, 0], emb[mask_resto, 1],
                c="#dddddd", s=18, label="outras", edgecolors="none")
    # depois as top-10 coloridas
    cores = cm.tab10(np.linspace(0, 1, len(top10)))
    for cor, cls in zip(cores, top10):
        mask = labels == cls
        nome = classnames[np.where(mask)[0][0]]
        plt.scatter(emb[mask, 0], emb[mask, 1], c=[cor],
                    s=55, edgecolors="black", linewidths=0.5,
                    label=f"{nome} ({mask.sum()})")
    plt.legend(loc="best", fontsize=9, framealpha=0.9)
    plt.title(titulo)
    plt.xlabel("dim 1"); plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()
    print(f"  -> {caminho}")


def main():
    hists = np.load(FEAT / f"bovw_hist_K{K_VW}.npy")
    labels = np.load(FEAT / "labels.npy")
    classnames = np.load(FEAT / "classnames.npy")
    print(f"Hists shape: {hists.shape}")

    print("Rodando UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb_umap = reducer.fit_transform(hists)
    plot_proj(emb_umap, labels, classnames,
              f"UMAP da BoVW (K={K_VW}) - top 10 classes coloridas",
              SAIDA / "visualizacao_bovw_umap.png")

    print("Rodando t-SNE...")
    tsne = TSNE(n_components=2, perplexity=20, random_state=42, init="pca")
    emb_tsne = tsne.fit_transform(hists)
    plot_proj(emb_tsne, labels, classnames,
              f"t-SNE da BoVW (K={K_VW}) - top 10 classes coloridas",
              SAIDA / "visualizacao_bovw_tsne.png")


if __name__ == "__main__":
    main()
