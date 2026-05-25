"""
Descritores de imagens para o Trabalho 3 - Processamento de Imagens.

5 descritores:
    1. desc_hsv       - histograma HSV (cor)
    2. desc_hog       - HOG (forma/gradientes)
    3. desc_lbp       - LBP uniforme (textura local)
    4. desc_orb       - ORB (keypoints, base do BoVW)
    5. desc_gabor     - banco de filtros de Gabor (textura, NAO visto em aula)

Gabor eh implementado manualmente (numpy + FFT no estilo da professora,
aula de restauracao) porque eh o descritor "nao visto em aula" exigido
pelo enunciado.
HSV usa skimage.color.rgb2hsv - mesma chamada que a professora usa no
notebook Aula17 (cores).
HOG, LBP e ORB usam skimage.feature - mesma forma do segmentacao/cod2.py.
"""

import numpy as np
import imageio.v3 as iio

from skimage.feature import hog, local_binary_pattern, ORB
from skimage import color


# -----------------------------------------------------------------------------
# Utilitarios (estilo da professora)
# -----------------------------------------------------------------------------

def carregar_imagem(path):
    """Carrega imagem garantindo formato RGB uint8 (H, W, 3)."""
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8)


def luminosity(img):
    """Funcao da professora: converte RGB para preto-e-branco."""
    return (0.21 * img[..., 0] + 0.72 * img[..., 1] + 0.07 * img[..., 2]).astype(np.uint8)


def _norm_l1(v, eps=1e-7):
    s = v.sum()
    return v / (s + eps)


# -----------------------------------------------------------------------------
# 1. Histograma HSV (cor)
# -----------------------------------------------------------------------------

def desc_hsv(img, bins=32):
    """
    Histograma concatenado dos 3 canais HSV, normalizado L1.
    Usa color.rgb2hsv (mesmo import do notebook Aula17 da professora).
    """
    hsv = color.rgb2hsv(img)
    h_hist, _ = np.histogram(hsv[..., 0], bins=bins, range=(0.0, 1.0))
    s_hist, _ = np.histogram(hsv[..., 1], bins=bins, range=(0.0, 1.0))
    v_hist, _ = np.histogram(hsv[..., 2], bins=bins, range=(0.0, 1.0))
    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    return _norm_l1(feat)


# -----------------------------------------------------------------------------
# 2. HOG (forma / gradientes)
# -----------------------------------------------------------------------------

def desc_hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9):
    gray = luminosity(img)
    feat = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


# -----------------------------------------------------------------------------
# 3. LBP uniforme (textura local)
# -----------------------------------------------------------------------------

def desc_lbp(img, P=8, R=1):
    gray = luminosity(img)
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    return _norm_l1(hist.astype(np.float32))


# -----------------------------------------------------------------------------
# 4. ORB (keypoints binarios - usado pelo BoVW)
# -----------------------------------------------------------------------------

def desc_orb(img, n_keypoints=200):
    """Retorna os descritores binarios brutos (K, 256) ou None se nao detectar nada."""
    gray = luminosity(img)
    orb = ORB(n_keypoints=n_keypoints)
    try:
        orb.detect_and_extract(gray)
        return orb.descriptors.astype(np.uint8)
    except (RuntimeError, IndexError):
        return None


# -----------------------------------------------------------------------------
# 5. Banco de Gabor (implementacao manual com FFT no estilo da professora)
# -----------------------------------------------------------------------------

def _gabor_kernel(frequency, theta, sigma=None, n_stds=3):
    """
    Constroi kernel de Gabor 2D: gaussiana modulada por cosseno/seno.
    Retorna (kernel_real, kernel_imag).

    g_real(x, y) = exp(-(xr^2 + yr^2)/(2*sigma^2)) * cos(2*pi*f*xr)
    g_imag(x, y) = exp(-(xr^2 + yr^2)/(2*sigma^2)) * sin(2*pi*f*xr)
    onde (xr, yr) sao (x, y) rotacionados por theta.
    """
    if sigma is None:
        sigma = 1.0 / frequency  # uma escolha simples e tradicional
    r = int(np.ceil(n_stds * sigma)) + 1
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2))
    kernel_real = envelope * np.cos(2 * np.pi * frequency * xr)
    kernel_imag = envelope * np.sin(2 * np.pi * frequency * xr)
    return kernel_real.astype(np.float32), kernel_imag.astype(np.float32)


def _pad_filter(kernel, img):
    """Pad o kernel ate o tamanho da imagem (estilo aulas/restauracao/cod2.md)."""
    h, w = img.shape
    hk, wk = kernel.shape
    pad_h_before = (h - hk) // 2
    pad_h_after = h - hk - pad_h_before
    pad_w_before = (w - wk) // 2
    pad_w_after = w - wk - pad_w_before
    return np.pad(kernel, ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)))


def _filter_fd(img, kernel):
    """Convolucao via FFT, igual ao filter_fd da aula de restauracao."""
    kernel_pad = _pad_filter(kernel, img)
    f_img = np.fft.fft2(img)
    f_kernel = np.fft.fft2(kernel_pad)
    return np.fft.fftshift(np.fft.ifft2(f_img * f_kernel)).real


def desc_gabor(
    img,
    frequencies=(0.1, 0.2, 0.3, 0.4),
    thetas=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
):
    """
    Para cada (frequencia, orientacao):
      1. constroi kernel de Gabor 2D (real e imag)
      2. convolui com a imagem em cinza via FFT
      3. tira modulo (magnitude) e guarda media e desvio padrao

    Saida: 2 * len(frequencies) * len(thetas) features.
    """
    gray = luminosity(img).astype(np.float32) / 255.0
    feats = []
    for f in frequencies:
        for t in thetas:
            kr, ki = _gabor_kernel(f, t)
            r = _filter_fd(gray, kr)
            i = _filter_fd(gray, ki)
            mag = np.sqrt(r ** 2 + i ** 2)
            feats.append(mag.mean())
            feats.append(mag.std())
    return np.array(feats, dtype=np.float32)


# -----------------------------------------------------------------------------
# Registry para facilitar iteracao
# -----------------------------------------------------------------------------

DESCRITORES_FIXOS = {
    "hsv": desc_hsv,
    "hog": desc_hog,
    "lbp": desc_lbp,
    "gabor": desc_gabor,
}
