"""
Pipeline de classificacao:
  1. carrega features pre-extraidas
  2. filtra classes com <3 imagens
  3. split estratificado manual 80/10/10 (manual porque sklearn nao lida bem
     com classes de 3-5 amostras)
  4. treina Random Forest para cada descritor isolado e algumas combinacoes
  5. salva tabela de acuracias e matriz de confusao da melhor combinacao

Saidas:
  resultados/classificacao_tabela.txt
  resultados/classificacao_matriz_<combo>.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


RAIZ = Path("/Users/leticia.neves/Desktop/6sem/2027/leo/entrega3")
FEAT = RAIZ / "features"
SAIDA = RAIZ / "resultados"
SAIDA.mkdir(exist_ok=True)

SEED = 42
MIN_POR_CLASSE = 3


def carregar_tudo():
    feats = {
        "hsv":   np.load(FEAT / "hsv.npy"),
        "hog":   np.load(FEAT / "hog.npy"),
        "lbp":   np.load(FEAT / "lbp.npy"),
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

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)
    acc_va = accuracy_score(y_va, clf.predict(X_va))
    pred_te = clf.predict(X_te)
    acc_te = accuracy_score(y_te, pred_te)

    # top-3 no teste
    probas = clf.predict_proba(X_te)
    top3 = np.argsort(probas, axis=1)[:, -3:]
    classes = clf.classes_
    acertou_top3 = np.array(
        [y_te[i] in classes[top3[i]] for i in range(len(y_te))]
    )
    acc_te_top3 = acertou_top3.mean()

    return {
        "acc_val": acc_va,
        "acc_test": acc_te,
        "acc_test_top3": acc_te_top3,
        "pred_test": pred_te,
        "y_test": y_te,
        "dim": X_tr.shape[1],
    }


def matriz_confusao_plot(y_true, y_pred, classnames_full, caminho, titulo):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # rotulos
    rot = []
    for c in classes:
        nome = classnames_full[np.where(classnames_full_idx == c)[0][0]] \
            if classnames_full_idx is not None else str(c)
        rot.append(nome)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(rot, rotation=90, fontsize=7)
    ax.set_yticklabels(rot, fontsize=7)
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    ax.set_title(titulo)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()


# variavel global preenchida em main (usada pelo plot)
classnames_full_idx = None


def main():
    global classnames_full_idx
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
        ("hog",),
        ("lbp",),
        ("gabor",),
        ("hsv", "hog"),
        ("hsv", "lbp"),
        ("hsv", "gabor"),
        ("hog", "lbp"),
        ("hog", "gabor"),
        ("lbp", "gabor"),
        ("hsv", "hog", "lbp"),
        ("hsv", "hog", "lbp", "gabor"),
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
                     f"acc={r['acc_test']:.3f}  top3={r['acc_test_top3']:.3f}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        out = SAIDA / f"classificacao_matriz_{nome_cfg.replace('+','_')}.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  -> {out}")

    linhas = ["combinacao | dim | acc_val | acc_test | acc_test_top3"]
    linhas.append("-" * 70)
    melhor = None
    descritores_isolados = {("hsv",), ("hog",), ("lbp",), ("gabor",)}
    for cfg in configs:
        r = treinar_avaliar(feats, labels, list(cfg), idx_tr, idx_va, idx_te)
        nome_cfg = "+".join(cfg)
        linhas.append(
            f"{nome_cfg:<30} | {r['dim']:>5} | {r['acc_val']:.3f}   | "
            f"{r['acc_test']:.3f}    | {r['acc_test_top3']:.3f}"
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
