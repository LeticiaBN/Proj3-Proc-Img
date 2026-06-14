"""
Extrai os 7 descritores em toda a base de pets e salva em features/.

Saida:
  features/hsv.npy        (N, D_hsv)         float32
  features/corr.npy       (N, D_corr)        float32
  features/hog.npy        (N, D_hog)         float32
  features/lbp.npy        (N, D_lbp)         float32
  features/glcm.npy       (N, D_glcm)        float32
  features/gabor.npy      (N, D_gabor)       float32
  features/orb.pkl        list of (K_i, 256) uint8, K_i varia por imagem
  features/labels.npy     (N,)               int    (class_id)
  features/classnames.npy (N,)               str    (nome do pet)
  features/filenames.npy  (N,)               str    (caminho relativo)
"""

import csv
import pickle
import time
from pathlib import Path

import numpy as np

from descritores import (
    carregar_imagem,
    desc_hsv, desc_correlograma, desc_hog, desc_lbp, desc_glcm, desc_gabor, desc_orb,
)


RAIZ = Path("/Users/leticia.neves/Desktop/6sem/2027/leo/entrega3")
PETS = RAIZ / "pets256"
CSV = RAIZ / "pets.csv"
SAIDA = RAIZ / "features"
SAIDA.mkdir(exist_ok=True)


def ler_csv():
    linhas = []
    with open(CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            cid, cname, fname = [c.strip() for c in row]
            linhas.append((int(cid), cname, fname))
    return linhas


def main():
    linhas = ler_csv()
    N = len(linhas)
    print(f"Total de imagens: {N}")

    hsv_feats, corr_feats, hog_feats, lbp_feats, glcm_feats, gabor_feats, orb_feats = [], [], [], [], [], [], []
    labels, classnames, filenames = [], [], []

    t0 = time.perf_counter()
    for i, (cid, cname, fname) in enumerate(linhas):
        img = carregar_imagem(PETS / fname)

        hsv_feats.append(desc_hsv(img))
        corr_feats.append(desc_correlograma(img))
        hog_feats.append(desc_hog(img))
        lbp_feats.append(desc_lbp(img))
        glcm_feats.append(desc_glcm(img))
        gabor_feats.append(desc_gabor(img))
        orb_feats.append(desc_orb(img))  # pode ser None

        labels.append(cid)
        classnames.append(cname)
        filenames.append(fname)

        if (i + 1) % 25 == 0 or i == N - 1:
            elap = time.perf_counter() - t0
            print(f"  {i+1}/{N}  ({elap:.1f}s, ~{elap/(i+1):.2f}s/img)")

    # empilha os descritores de tamanho fixo
    X_hsv = np.stack(hsv_feats).astype(np.float32)
    X_corr = np.stack(corr_feats).astype(np.float32)
    X_hog = np.stack(hog_feats).astype(np.float32)
    X_lbp = np.stack(lbp_feats).astype(np.float32)
    X_glcm = np.stack(glcm_feats).astype(np.float32)
    X_gabor = np.stack(gabor_feats).astype(np.float32)

    np.save(SAIDA / "hsv.npy", X_hsv)
    np.save(SAIDA / "corr.npy", X_corr)
    np.save(SAIDA / "hog.npy", X_hog)
    np.save(SAIDA / "lbp.npy", X_lbp)
    np.save(SAIDA / "glcm.npy", X_glcm)
    np.save(SAIDA / "gabor.npy", X_gabor)

    # ORB tem K variavel -> pickle
    with open(SAIDA / "orb.pkl", "wb") as f:
        pickle.dump(orb_feats, f)

    np.save(SAIDA / "labels.npy", np.array(labels, dtype=np.int32))
    np.save(SAIDA / "classnames.npy", np.array(classnames))
    np.save(SAIDA / "filenames.npy", np.array(filenames))

    # relatorio rapido
    n_orb_ok = sum(1 for x in orb_feats if x is not None)
    n_kp_medio = np.mean([len(x) for x in orb_feats if x is not None])

    print("\n=== Dimensoes ===")
    print(f"  HSV   : {X_hsv.shape}")
    print(f"  Corr  : {X_corr.shape}")
    print(f"  HOG   : {X_hog.shape}")
    print(f"  LBP   : {X_lbp.shape}")
    print(f"  GLCM  : {X_glcm.shape}")
    print(f"  Gabor : {X_gabor.shape}")
    print(f"  ORB   : {n_orb_ok}/{N} imagens com keypoints, media {n_kp_medio:.0f} kp/img")
    print(f"\nArquivos salvos em {SAIDA}/")


if __name__ == "__main__":
    main()
