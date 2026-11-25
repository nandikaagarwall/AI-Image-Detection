import numpy as np
from scipy.ndimage import gaussian_filter, laplace, sobel
from scipy.stats import kurtosis, skew
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import hog, local_binary_pattern

N_RADIAL = 16

def _hist(ch, bins, rng=(0, 1)):
    h, _ = np.histogram(ch, bins=bins, range=rng)
    return h / max(ch.size, 1)

def extract_features(img):
    """img: uint8 array (H, W, 3) -> 1-D float32 feature vector."""
    img = np.asarray(img, dtype=np.uint8)
    f = img.astype(np.float32) / 255.0
    gray = rgb2gray(f)
    hsv = rgb2hsv(f)
    feats = []

    # colour
    for c in range(3):
        feats.append(_hist(f[..., c], 16))
    feats.append(_hist(hsv[..., 0], 16)); feats.append(_hist(hsv[..., 1], 8)); feats.append(_hist(hsv[..., 2], 8))
    feats.append(f.mean(axis=(0, 1))); feats.append(f.std(axis=(0, 1)))

    # HOG
    feats.append(hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True))

    # LBP (uniform, P=8, R=1 -> 10 bins)
    lbp = local_binary_pattern((gray * 255).astype(np.uint8), 8, 1, method='uniform')
    h, _ = np.histogram(lbp, bins=10, range=(0, 10))
    feats.append(h / lbp.size)

    # FFT radial spectrum
    F = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    mag = np.log1p(np.abs(F))
    H, W = gray.shape
    yy, xx = np.indices((H, W))
    r = np.hypot(yy - H / 2, xx - W / 2)
    rb = np.minimum((r / (r.max() + 1e-6) * N_RADIAL).astype(int), N_RADIAL - 1)
    radial = np.bincount(rb.ravel(), weights=mag.ravel(), minlength=N_RADIAL) / np.maximum(np.bincount(rb.ravel(), minlength=N_RADIAL), 1)
    feats.append(radial)
    power = np.abs(F) ** 2
    feats.append([power[r > r.max() * 0.5].sum() / (power.sum() + 1e-8)])

    # noise residual + edges
    res = gray - gaussian_filter(gray, 1.0)
    feats.append([res.std(), np.abs(res).mean(), kurtosis(res.ravel()), skew(res.ravel())])
    feats.append([laplace(gray).var(), np.hypot(sobel(gray, 0), sobel(gray, 1)).mean()])

    # entropy
    g = np.histogram(gray, bins=64, range=(0, 1))[0] / gray.size
    g = g[g > 0]
    feats.append([-(g * np.log2(g)).sum()])

    out = np.concatenate([np.ravel(np.asarray(x, dtype=np.float32)) for x in feats])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def extract_batch(imgs):
    return np.stack([extract_features(i) for i in imgs])
