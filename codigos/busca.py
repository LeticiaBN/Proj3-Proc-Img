"""
Tarefa de busca por distancia euclidiana.

Para cada descritor fixo (HSV, HOG, LBP, Gabor) e algumas combinacoes,
calcula matriz de distancia par a par e:
  - reporta precision@1, precision@5 medias (so para imagens de classes
    com mais de uma foto)
  - gera figura com top-5 vizinhos para 4 imagens-consulta

Saidas:
  resultados/busca_precisao.txt
  resultados/busca_top5_<combo>.png
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio


# caminho padrao da entrega; pode ser sobrescrito por env var PROJ3_RAIZ
RAIZ = Path(os.environ.get("PROJ3_RAIZ", "/Users/leticia.neves/Desktop/6sem/2027/leo/entrega3"))
PETS = RAIZ / "pets256"
FEAT = RAIZ / "features"
SAIDA = RAIZ / "resultados"
SAIDA.mkdir(exist_ok=True)


def carregar():
    feats = {
        "hsv":   np.load(FEAT / "hsv.npy"),
        "corr":  np.load(FEAT / "corr.npy"),
        "hog":   np.load(FEAT / "hog.npy"),
        "lbp":   np.load(FEAT / "lbp.npy"),
        "glcm":  np.load(FEAT / "glcm.npy"),
        "gabor": np.load(FEAT / "gabor.npy"),
    }
    labels = np.load(FEAT / "labels.npy")
    classnames = np.load(FEAT / "classnames.npy")
    filenames = np.load(FEAT / "filenames.npy")
    return feats, labels, classnames, filenames


def zscore(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-7
    return (X - mu) / sd


def concat_norm(feats, nomes):
    """z-score por descritor e concatena (evita HOG dominar pela escala)."""
    return np.concatenate([zscore(feats[n]) for n in nomes], axis=1)


def pairwise_euclid(X):
    sq = (X ** 2).sum(1, keepdims=True)
    d2 = sq + sq.T - 2 * X @ X.T
    np.fill_diagonal(d2, 0)
    return np.sqrt(np.maximum(d2, 0))


def precision_at_k(D, labels, k_list=(1, 5)):
    """
    Media de precision@k e mAP considerando apenas imagens-consulta cujas
    classes tem >=2 fotos (senao nao da pra acertar).

    mAP = media da Average Precision por consulta, onde a AP percorre o ranking
    completo e media as precisoes nos pontos em que aparece um item relevante
    (mesma classe), normalizando pelo numero de relevantes. E a metrica padrao
    de retrieval, mais robusta que P@k para classes de tamanho variavel.
    """
    N = len(labels)
    D2 = D.copy()
    np.fill_diagonal(D2, np.inf)
    rank = np.argsort(D2, axis=1)  # (N, N) ordenado por distancia
    out = {}
    uniq, counts = np.unique(labels, return_counts=True)
    classes_com_par = set(uniq[counts >= 2].tolist())
    valid = np.array([l in classes_com_par for l in labels])
    for k in k_list:
        top = rank[:, :k]
        same = (labels[top] == labels[:, None])  # (N, k)
        prec = same.mean(axis=1)  # precisao por consulta
        out[k] = prec[valid].mean()

    # mAP
    aps = []
    for q in range(N):
        if not valid[q]:
            continue
        order = rank[q]
        order = order[order != q]                      # remove a propria consulta
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
    if axes.ndim == 1:
        axes = axes[None, :]
    fig.suptitle(titulo + "\nVerde = mesma classe  |  Vermelho = diferente",
                 fontsize=13, y=0.995)

    for qi, q in enumerate(queries):
        d = D[q].copy()
        d[q] = np.inf
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
            acerto = labels[idx] == labels[q]
            cor = "#2ca02c" if acerto else "#d62728"
            marca = "OK" if acerto else "X"
            ax.set_title(f"top {col} [{marca}]\n{classnames[idx]}",
                         fontsize=9, color=cor, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(cor); s.set_linewidth(3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  -> {caminho}")


def main():
    feats, labels, classnames, filenames = carregar()
    N = len(labels)
    print(f"N={N}, descritores: {list(feats)}")

    configs = [
        ("hsv",),
        ("corr",),
        ("hog",),
        ("lbp",),
        ("glcm",),
        ("gabor",),
        ("hsv", "corr"),
        ("hsv", "glcm"),
        ("hsv", "gabor"),
        ("hsv", "glcm", "gabor"),
        ("hsv", "corr", "glcm"),
        ("hsv", "lbp", "glcm", "gabor"),
        ("hsv", "corr", "lbp", "glcm", "gabor"),
        ("hsv", "hog", "lbp", "glcm", "corr", "gabor"),
    ]

    # escolhe 4 queries de classes com >=5 imagens
    rng = np.random.default_rng(42)
    classes_grandes = {}
    for i, c in enumerate(labels):
        classes_grandes.setdefault(int(c), []).append(i)
    candidatas = [v[0] for v in classes_grandes.values() if len(v) >= 5]
    queries = rng.choice(candidatas, size=4, replace=False).tolist()

    linhas = ["combinacao | dim | mAP | P@1 | P@5"]
    linhas.append("-" * 56)

    for cfg in configs:
        X = concat_norm(feats, list(cfg))
        D = pairwise_euclid(X)
        p = precision_at_k(D, labels, k_list=(1, 5))
        nome = "+".join(cfg)
        linhas.append(f"{nome:<28} | {X.shape[1]:>5} | {p['mAP']:.3f} | {p[1]:.3f} | {p[5]:.3f}")
        out = SAIDA / f"busca_top5_{nome.replace('+','_')}.png"
        plot_top5(D, queries, labels, classnames, filenames, out,
                  titulo=f"Busca - {nome}")

    txt = "\n".join(linhas)
    print("\n" + txt)
    (SAIDA / "busca_precisao.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
