"""Filter-bank CSP features (CPU reference), the family of the BCI Competition IV 2a winner (Ang et al.).

Used by the session-T model selection only (``real_benchmark select2``); it has no GPU path. Per band the
trace-normalised window covariances are projected with one-vs-rest CSP filters learnt on the training windows,
and the log variance of every projection is a feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import torch
from scipy.signal import firwin

FB_BANDS = tuple((lo, lo + 4) for lo in range(4, 40, 4))      # 4-8, 8-12, ..., 36-40 Hz
FB_TAPS = 151
CSP_PAIRS = 2                                                  # m: 2m filters per class and band


def fb_taps(fs: float = 250.0) -> list[np.ndarray]:
    return [firwin(FB_TAPS, [lo, hi], pass_zero=False, fs=fs, window="hamming") for lo, hi in FB_BANDS]


def trace_normalised_cov(y: np.ndarray) -> np.ndarray:
    """``y (B, C, T)`` (mean-removed) -> ``(B, C, C)`` with trace 1."""
    cov = y @ y.transpose(0, 2, 1) / (y.shape[2] - 1)
    return cov / np.trace(cov, axis1=1, axis2=2)[:, None, None]


def csp_ovr(covs: np.ndarray, y: np.ndarray, n_classes: int, m: int = CSP_PAIRS) -> np.ndarray:
    """One-vs-rest CSP filters, ``(n_classes * 2m, C)``: for class c the generalized eigenvectors of
    (Sigma_c, Sigma_c + Sigma_rest) with the m largest and m smallest eigenvalues (Sigma = class-mean covariance)."""
    class_cov = np.stack([covs[y == c].mean(axis=0) for c in range(n_classes)])
    rows = []
    for c in range(n_classes):
        a = class_cov[c]
        rest = np.delete(class_cov, c, axis=0).mean(axis=0)
        _, vecs = scipy.linalg.eigh(a, a + rest)                # ascending eigenvalues, columns = filters
        rows.append(np.hstack([vecs[:, :m], vecs[:, -m:]]).T)
    return np.vstack(rows)


def log_variance(covs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``diag(W C W^T)`` per window, log-transformed: ``(B, n_filters)``."""
    return np.log(np.einsum("fi,nij,fj->nf", w, covs, w))


class FBCSP:
    """Fit CSP filters per band on training covariances; ``transform`` gives the stacked log-variance features."""

    def __init__(self, n_classes: int = 4, m: int = CSP_PAIRS) -> None:
        self.n_classes, self.m = n_classes, m
        self.filters: list[np.ndarray] = []

    def fit(self, covs_per_band: list[np.ndarray], y: np.ndarray) -> "FBCSP":
        self.filters = [csp_ovr(c, y, self.n_classes, self.m) for c in covs_per_band]
        return self

    def transform(self, covs_per_band: list[np.ndarray]) -> np.ndarray:
        return np.hstack([log_variance(c, w) for c, w in zip(covs_per_band, self.filters)])


# =========================================================================== deployable decoder
@dataclass(frozen=True)
class FBCSPParams:
    """Filter-bank CSP + LDA: what :class:`FBCSPLDA` and ``GPUPipeline`` need (same role as ``RiemannianLDAParams``)."""
    taps: np.ndarray            # (n_bands, K) FIR taps, one row per band
    filters: np.ndarray         # (n_bands, n_filters, C) CSP filters
    select: np.ndarray          # (d,) indices into the flattened (n_bands * n_filters) log-variance vector
    lda_w: np.ndarray           # (n_classes, d)
    lda_b: np.ndarray           # (n_classes,)
    class_names: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return self.lda_w.shape[1]

    @property
    def n_channels(self) -> int:
        return self.filters.shape[2]


def _band_covs(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Windows ``(B, T, C)`` -> trace-normalised covariances of the band-passed windows ``(B, C, C)``."""
    from .filters import bandpass_cpu
    return trace_normalised_cov(bandpass_cpu(x, taps))


def features_cpu(params: FBCSPParams, x: np.ndarray) -> dict[str, np.ndarray]:
    """CPU reference for windows ``x (B, T, C)``: selected log-variance features and LDA scores."""
    covs = [_band_covs(x, t) for t in params.taps]
    feat = np.hstack([log_variance(c, w) for c, w in zip(covs, params.filters)])[:, params.select]
    return {"feat": feat, "scores": feat @ params.lda_w.T + params.lda_b}


def fit_fbcsp(x: np.ndarray, y: np.ndarray, fs: float = 250.0, d: int = 48,
              class_names: tuple[str, ...] | None = None) -> FBCSPParams:
    """Offline CPU training on windows ``x (N, T, C)``: CSP per band, mutual-information selection of ``d`` features,
    LDA (automatic shrinkage). The same recipe as candidate F in the session-T selection (SPEC.md section 13.1)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.feature_selection import mutual_info_classif
    taps = np.stack(fb_taps(fs))
    n_classes = int(y.max()) + 1
    covs = [_band_covs(x, t) for t in taps]
    model = FBCSP(n_classes).fit(covs, y)
    feat = model.transform(covs)
    order = np.argsort(-mutual_info_classif(feat, y, n_neighbors=3, random_state=0))[:min(d, feat.shape[1])]
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(feat[:, order], y)
    w, b = lda.coef_, lda.intercept_
    if w.shape[0] == 1:
        w, b = np.vstack([-w / 2, w / 2]), np.array([-b[0] / 2, b[0] / 2])
    names = class_names if class_names is not None else tuple(f"Class {i}" for i in range(w.shape[0]))
    return FBCSPParams(taps, np.stack(model.filters), order.astype(np.int64), w, b, names)


class FBCSPLDA(torch.nn.Module):
    """Batched float64 CSP log-variance features -> LDA on any torch device. No eigendecomposition at run time, so
    nothing here synchronises the host. Same four-step interface as ``RiemannianLDA`` (covariance, whiten,
    log_features, head) so ``GPUPipeline`` can drive either."""

    def __init__(self, params: FBCSPParams, device: torch.device | str = "cpu") -> None:
        super().__init__()
        f64 = dict(dtype=torch.float64)
        self.n_channels = params.filters.shape[2]
        self.register_buffer("csp", torch.from_numpy(params.filters).to(**f64))          # (nb, nf, C)
        self.register_buffer("select", torch.from_numpy(params.select).long())
        self.lda = torch.nn.Linear(params.n_features, params.lda_w.shape[0], dtype=torch.float64)
        with torch.no_grad():
            self.lda.weight.copy_(torch.from_numpy(params.lda_w))
            self.lda.bias.copy_(torch.from_numpy(params.lda_b))
        self.to(device).eval()

    def covariance(self, y: torch.Tensor) -> torch.Tensor:
        """``y (B, nb, C, T)`` band-passed, mean-removed -> ``(B, nb, C, C)`` float64, trace 1."""
        y = y.to(torch.float64)
        cov = y @ y.transpose(-1, -2) / (y.shape[-1] - 1)
        return cov / cov.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]

    def whiten(self, cov: torch.Tensor) -> torch.Tensor:
        """No reference point in this decoder: identity (keeps the pipeline's stage layout)."""
        return cov

    def log_features(self, cov: torch.Tensor) -> torch.Tensor:
        """``(B, nb, C, C)`` -> selected log variances of the CSP projections ``(B, d)``."""
        var = torch.einsum("nfi,bnij,nfj->bnf", self.csp, cov, self.csp)
        return torch.log(var).flatten(1)[:, self.select]

    def head(self, feat: torch.Tensor) -> torch.Tensor:
        return self.lda(feat)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.head(self.log_features(self.whiten(self.covariance(y))))
