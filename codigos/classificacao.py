"""
Pipeline de classificacao:
  1. carrega features pre-extraidas
  2. filtra classes com <3 imagens
  3. split estratificado manual 80/10/10 (manual porque sklearn nao lida bem
     com classes de 3-5 amostras)
  4. treina SVM (selecao de kernel/C na validacao) para cada descritor
     isolado e algumas combinacoes
  5. salva tabela de acuracias e matriz de confusao da melhor combinacao

Saidas:
  resultados/classificacao_tabela.txt
  resultados/classificacao_matriz_<combo>.png
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# grade de hiperparametros da SVM, selecionada na validacao
# (mesma ideia vista no slide "Pipeline Classica" da professora: SVM como
#  algoritmo de classificacao; kernel e C escolhidos por validacao)
SVM_GRID = (
    [{"kernel": "linear", "C": c} for c in (0.1, 1, 10)]
    + [{"kernel": "rbf", "C": c, "gamma": g}
       for c in (0.1, 1, 10) for g in ("scale", 0.01, 0.001)]
)


# caminho padrao da entrega; pode ser sobrescrito por env var PROJ3_RAIZ
RAIZ = Path(os.environ.get("PROJ3_RAIZ", "/Users/leticia.neves/Desktop/6sem/2027/leo/entrega3"))
FEAT = RAIZ / "features"
SAIDA = RAIZ / "resultados"
SAIDA.mkdir(exist_ok=True)

SEED = 42
MIN_POR_CLASSE = 3


def carregar_tudo():
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
    return feats, labels, classnames


def filtrar_classes(feats, labels, classnames, min_count=MIN_POR_CLASSE):
    uniq, counts = np.unique(labels, return_counts=True)
    validas = set(uniq[counts >= min_count].tolist())
    mask = np.array([y in validas for y in labels])
    print(f"Classes antes do filtro: {len(uniq)}")
    print(f"Classes com >={min_count} imagens: {len(validas)}")
    print(f"Imagens mantidas: {mask.sum()}/{len(labels)}")
    feats_f = {k: v[mask] for k, v in feats.items()}
    return feats_f, labels[mask], classnames[mask]


def split_80_10_10(labels, seed=SEED):
    """
    Split estratificado por classe.
    Para cada classe com N amostras:
      - test = max(1, round(0.1 * N))
      - val  = max(1, round(0.1 * N))    se N >= 3
      - train = N - test - val
    """
    rng = np.random.default_rng(seed)
    idx_train, idx_val, idx_test = [], [], []
    for cls in np.unique(labels):
        idxs = np.where(labels == cls)[0]
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, round(0.1 * n))
        n_val = max(1, round(0.1 * n))
        # garante que sobre pelo menos 1 pra treino
        while n_test + n_val >= n:
            if n_val > 0:
                n_val -= 1
            else:
                n_test -= 1
        idx_test.extend(idxs[:n_test].tolist())
        idx_val.extend(idxs[n_test:n_test + n_val].tolist())
        idx_train.extend(idxs[n_test + n_val:].tolist())
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def montar(feats, nomes, idxs, scaler=None, fit=False):
    """Concatena descritores escolhidos, opcionalmente padroniza."""
    blocos = [feats[n][idxs] for n in nomes]
    X = np.concatenate(blocos, axis=1)
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, scaler


def treinar_avaliar(feats, labels, nomes, idx_tr, idx_va, idx_te):
    X_tr, sc = montar(feats, nomes, idx_tr, fit=True)
    X_va, _ = montar(feats, nomes, idx_va, scaler=sc)
    X_te, _ = montar(feats, nomes, idx_te, scaler=sc)
    y_tr, y_va, y_te = labels[idx_tr], labels[idx_va], labels[idx_te]

    # seleciona kernel/C pela acuracia de validacao
    melhor_clf, melhor_acc_va, melhor_params = None, -1.0, None
    for params in SVM_GRID:
        clf = SVC(
            class_weight="balanced",
            decision_function_shape="ovr",
            random_state=SEED,
            **params,
        )
        clf.fit(X_tr, y_tr)
        acc_va = accuracy_score(y_va, clf.predict(X_va))
        if acc_va > melhor_acc_va:
            melhor_acc_va, melhor_clf, melhor_params = acc_va, clf, params

    clf = melhor_clf
    acc_va = melhor_acc_va
    pred_te = clf.predict(X_te)
    acc_te = accuracy_score(y_te, pred_te)
    f1_te = f1_score(y_te, pred_te, average="weighted", zero_division=0)

    # top-3 no teste: ranqueia pelas margens da decisao one-vs-rest
    scores = clf.decision_function(X_te)
    classes = clf.classes_
    if scores.ndim == 1:  # caso binario (2 classes apos o filtro)
        scores = np.column_stack([-scores, scores])
    top3 = np.argsort(scores, axis=1)[:, -3:]
    acertou_top3 = np.array(
        [y_te[i] in classes[top3[i]] for i in range(len(y_te))]
    )
    acc_te_top3 = acertou_top3.mean()

    # string curta do modelo escolhido (para a tabela)
    if melhor_params["kernel"] == "linear":
        modelo = f"linear C={melhor_params['C']}"
    else:
        modelo = f"rbf C={melhor_params['C']} g={melhor_params['gamma']}"

    return {
        "acc_val": acc_va,
        "acc_test": acc_te,
        "f1_test": f1_te,
        "acc_test_top3": acc_te_top3,
        "modelo": modelo,
        "pred_test": pred_te,
        "y_test": y_te,
        "dim": X_tr.shape[1],
    }


def main():
    feats, labels, classnames = carregar_tudo()
    feats, labels, classnames = filtrar_classes(feats, labels, classnames)

    # mapa class_id -> nome (qualquer indice de cada classe serve, todos tem mesmo nome)
    uniq_labels = np.unique(labels)
    nome_por_id = {c: classnames[np.where(labels == c)[0][0]] for c in uniq_labels}

    idx_tr, idx_va, idx_te = split_80_10_10(labels)
    print(f"\nSplit: train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)}")

    # configuracoes a avaliar
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
        ("hsv", "corr", "glcm", "gabor"),
        ("hsv", "lbp", "glcm", "gabor"),
        ("hsv", "corr", "lbp", "glcm", "gabor"),
        ("hsv", "hog", "lbp", "glcm", "corr", "gabor"),
    ]

    def salvar_matriz(r, nome_cfg):
        y_true_nomes = np.array([nome_por_id[c] for c in r["y_test"]])
        y_pred_nomes = np.array([nome_por_id[c] for c in r["pred_test"]])
        cm_classes = sorted(set(y_true_nomes.tolist()) | set(y_pred_nomes.tolist()))
        cm = confusion_matrix(y_true_nomes, y_pred_nomes, labels=cm_classes)
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(cm_classes))); ax.set_yticks(range(len(cm_classes)))
        ax.set_xticklabels(cm_classes, rotation=90, fontsize=7)
        ax.set_yticklabels(cm_classes, fontsize=7)
        ax.set_xlabel("Predito"); ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusao (teste) - {nome_cfg}\n"
                     f"acc={r['acc_test']:.3f}  f1={r['f1_test']:.3f}  "
                     f"top3={r['acc_test_top3']:.3f}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        out = SAIDA / f"classificacao_matriz_{nome_cfg.replace('+','_')}.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  -> {out}")

    linhas = ["combinacao | dim | modelo | acc_val | acc_test | f1_test | acc_test_top3"]
    linhas.append("-" * 92)
    melhor = None
    descritores_isolados = {("hsv",), ("corr",), ("hog",), ("lbp",), ("glcm",), ("gabor",)}
    for cfg in configs:
        r = treinar_avaliar(feats, labels, list(cfg), idx_tr, idx_va, idx_te)
        nome_cfg = "+".join(cfg)
        linhas.append(
            f"{nome_cfg:<24} | {r['dim']:>5} | {r['modelo']:<18} | {r['acc_val']:.3f}   | "
            f"{r['acc_test']:.3f}    | {r['f1_test']:.3f}   | {r['acc_test_top3']:.3f}"
        )
        if melhor is None or r["acc_val"] > melhor[1]["acc_val"]:
            melhor = (nome_cfg, r)
        # salva matriz para cada descritor isolado
        if cfg in descritores_isolados:
            salvar_matriz(r, nome_cfg)

    txt = "\n".join(linhas)
    print("\n" + txt)
    (SAIDA / "classificacao_tabela.txt").write_text(txt + "\n")

    # matriz de confusao da melhor combinacao (por acc de validacao)
    print(f"\nMelhor por validacao: {melhor[0]}")
    salvar_matriz(melhor[1], melhor[0])


if __name__ == "__main__":
    main()
