"""
Bag of Visual Words com ORB.

  1. junta todos descritores ORB da base, subamostra
  2. KMeans (MiniBatch) constroi vocabulario visual de K palavras
  3. para cada imagem, atribui cada keypoint a palavra mais proxima e
     monta histograma normalizado de K bins
  4. busca por distancia euclidiana entre histogramas
  5. salva tudo em features/bovw_hist_K<K>.npy e features/bovw_vocab_K<K>.npy

Saidas:
  features/bovw_hist_K128.npy
  features/bovw_vocab_K128.npy
  resultados/bovw_precisao.txt
  resultados/bovw_top5.png
"""

import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from sklearn.cluster import MiniBatchKMeans


# caminho padrao da entrega; pode ser sobrescrito por env var PROJ3_RAIZ
RAIZ = Path(os.environ.get("PROJ3_RAIZ", "/Users/leticia.neves/Desktop/6sem/2027/leo/entrega3"))
PETS = RAIZ / "pets256"
FEAT = RAIZ / "features"
SAIDA = RAIZ / "resultados"
SAIDA.mkdir(exist_ok=True)

K_VW = 128
SEED = 42
MAX_DESC_PARA_VOCAB = 30000


def carregar_orb():
    with open(FEAT / "orb.pkl", "rb") as f:
        return pickle.load(f)


def construir_vocab(orb_lista, K=K_VW, max_amostras=MAX_DESC_PARA_VOCAB):
    todos = [d for d in orb_lista if d is not None]
    pilhado = np.concatenate(todos, axis=0).astype(np.float32)
    print(f"Total de descritores ORB: {pilhado.shape}")
    if pilhado.shape[0] > max_amostras:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(pilhado.shape[0], size=max_amostras, replace=False)
        amostra = pilhado[idx]
        print(f"Subamostrado para: {amostra.shape}")
    else:
        amostra = pilhado
    km = MiniBatchKMeans(
        n_clusters=K, random_state=SEED, batch_size=2048, n_init=5, max_iter=200,
    )
    km.fit(amostra)
    return km


def hist_bovw(orb_lista, km):
    K = km.n_clusters
    hists = np.zeros((len(orb_lista), K), dtype=np.float32)
    for i, d in enumerate(orb_lista):
        if d is None or len(d) == 0:
            continue
        ass = km.predict(d.astype(np.float32))
        h = np.bincount(ass, minlength=K).astype(np.float32)
        h /= (h.sum() + 1e-7)
        hists[i] = h
    return hists


def pairwise_euclid(X):
    sq = (X ** 2).sum(1, keepdims=True)
    d2 = sq + sq.T - 2 * X @ X.T
    np.fill_diagonal(d2, 0)
    return np.sqrt(np.maximum(d2, 0))


def precision_at_k(D, labels, k_list=(1, 5)):
    N = len(labels)
    D2 = D.copy(); np.fill_diagonal(D2, np.inf)
    rank = np.argsort(D2, axis=1)
    uniq, counts = np.unique(labels, return_counts=True)
    com_par = set(uniq[counts >= 2].tolist())
    valid = np.array([l in com_par for l in labels])
    out = {}
    for k in k_list:
        top = rank[:, :k]
        same = (labels[top] == labels[:, None])
        out[k] = same.mean(1)[valid].mean()
    # mAP (Average Precision media sobre consultas validas)
    aps = []
    for q in range(N):
        if not valid[q]:
            continue
        order = rank[q]
        order = order[order != q]
        rel = (labels[order] == labels[q]).astype(float)
        n_rel = rel.sum()
        if n_rel == 0:
            continue
        cum = np.cumsum(rel)
        prec_at = cum / (np.arange(len(rel)) + 1)
        aps.append((prec_at * rel).sum() / n_rel)
    out["mAP"] = float(np.mean(aps)) if aps else 0.0
    return out


def plot_top5(D, queries, labels, classnames, filenames, caminho, titulo, k=5):
    n_q = len(queries)
    fig, axes = plt.subplots(n_q, k + 1, figsize=(2.2 * (k + 1), 2.4 * n_q))
    if axes.ndim == 1: axes = axes[None, :]
    fig.suptitle(titulo + "\nVerde = mesma classe  |  Vermelho = diferente",
                 fontsize=13, y=0.995)
    for qi, q in enumerate(queries):
        d = D[q].copy(); d[q] = np.inf
        top = np.argsort(d)[:k]
        ax = axes[qi, 0]
        ax.imshow(iio.imread(PETS / filenames[q]))
        ax.set_title(f"CONSULTA\n{classnames[q]}", fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("black"); s.set_linewidth(2)
        for col, idx in enumerate(top, start=1):
            ax = axes[qi, col]
            ax.imshow(iio.imread(PETS / filenames[idx]))
            ok = labels[idx] == labels[q]
            cor = "#2ca02c" if ok else "#d62728"
            mk = "OK" if ok else "X"
            ax.set_title(f"top {col} [{mk}]\n{classnames[idx]}",
                         fontsize=9, color=cor, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(cor); s.set_linewidth(3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  -> {caminho}")


def main():
    orb_lista = carregar_orb()
    labels = np.load(FEAT / "labels.npy")
    classnames = np.load(FEAT / "classnames.npy")
    filenames = np.load(FEAT / "filenames.npy")

    km = construir_vocab(orb_lista, K=K_VW)
    np.save(FEAT / f"bovw_vocab_K{K_VW}.npy", km.cluster_centers_)
    print(f"Vocabulario salvo (K={K_VW})")

    hists = hist_bovw(orb_lista, km)
    np.save(FEAT / f"bovw_hist_K{K_VW}.npy", hists)
    print(f"Histogramas BoVW: shape {hists.shape}")

    D = pairwise_euclid(hists)
    p = precision_at_k(D, labels, k_list=(1, 5))
    txt = f"BoVW (K={K_VW})  mAP={p['mAP']:.3f}  P@1={p[1]:.3f}  P@5={p[5]:.3f}"
    print("\n" + txt)
    (SAIDA / "bovw_precisao.txt").write_text(txt + "\n")

    rng = np.random.default_rng(42)
    classes_g = {}
    for i, c in enumerate(labels):
        classes_g.setdefault(int(c), []).append(i)
    cand = [v[0] for v in classes_g.values() if len(v) >= 5]
    queries = rng.choice(cand, size=4, replace=False).tolist()
    plot_top5(D, queries, labels, classnames, filenames,
              SAIDA / "bovw_top5.png", titulo=f"Busca BoVW (K={K_VW})")


if __name__ == "__main__":
    main()
