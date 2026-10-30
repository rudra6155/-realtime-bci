"""Riemannian tangent-space + LDA classifier.

* CPU reference and offline training: numpy / scipy / scikit-learn, float64.
* GPU/torch module (:class:`RiemannianLDA`): covariance -> tangent space -> LDA, batched,
  float64, no Python loop over channels. ``torch.linalg.eigh`` is the one unavoidable host
  synchronization on CUDA (PyTorch documents it; there is no ``eigh_ex``).
* ONNX: the LDA head (a plain ``Linear``) is exported and run with onnxruntime.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import CLASS_NAMES, COV_SHRINKAGE
from .filters import bandpass_cpu, design_bandpass_taps


# =========================================================================== CPU reference
def covariance_cpu(y: np.ndarray, shrinkage: float = COV_SHRINKAGE) -> np.ndarray:
    """``y (B, C, T)`` mean-removed float64 -> ``(B, C, C)``: sample covariance, scaled so
    ``trace == C``, then shrunk toward the identity."""
    _, c, t = y.shape
    cov = y @ y.transpose(0, 2, 1) / (t - 1)
    trace = np.trace(cov, axis1=1, axis2=2)[:, None, None]
    cov = cov * (c / trace)
    return (1.0 - shrinkage) * cov + shrinkage * np.eye(c)


def _sym_fn(m: np.ndarray, fn) -> np.ndarray:
    w, v = scipy.linalg.eigh(m)
    return (v * fn(w)) @ v.T


def riemann_mean(covs: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """Affine-invariant (Karcher) mean of SPD matrices by fixed-point iteration."""
    mean = covs.mean(axis=0)
    for _ in range(max_iter):
        inv_sqrt = _sym_fn(mean, lambda w: w ** -0.5)
        sqrt = _sym_fn(mean, np.sqrt)
        step = np.mean([_sym_fn(inv_sqrt @ c @ inv_sqrt, np.log) for c in covs], axis=0)
        mean = sqrt @ _sym_fn(step, np.exp) @ sqrt
        if np.linalg.norm(step) < tol:
            break
    return mean


def _triu_spec(c: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-major upper-triangle indices and weights (off-diagonals x sqrt(2))."""
    r, col = np.triu_indices(c)
    return r * c + col, np.where(r == col, 1.0, np.sqrt(2.0))


def tangent_cpu(covs: np.ndarray, ref_inv_sqrt: np.ndarray) -> np.ndarray:
    """``(B, C, C)`` SPD -> ``(B, C(C+1)/2)`` tangent-space features at the reference."""
    c = covs.shape[-1]
    flat_idx, weights = _triu_spec(c)
    out = np.empty((covs.shape[0], len(flat_idx)))
    for i, cov in enumerate(covs):
        logm = _sym_fn(ref_inv_sqrt @ cov @ ref_inv_sqrt, np.log)
        out[i] = logm.reshape(-1)[flat_idx] * weights
    return out


@dataclass(frozen=True)
class RiemannianLDAParams:
    taps: np.ndarray            # (K,) FIR taps
    shrinkage: float
    ref_inv_sqrt: np.ndarray    # (C, C) C_ref^{-1/2}
    lda_w: np.ndarray           # (n_classes, F)
    lda_b: np.ndarray           # (n_classes,)
    class_names: tuple[str, ...] = CLASS_NAMES

    @property
    def n_features(self) -> int:
        return self.lda_w.shape[1]


def features_cpu(params: RiemannianLDAParams, x: np.ndarray) -> dict[str, np.ndarray]:
    """Full CPU reference for windows ``x (B, T, C)``: filtered, covariance, features, scores."""
    filt = bandpass_cpu(x, params.taps)
    cov = covariance_cpu(filt, params.shrinkage)
    feat = tangent_cpu(cov, params.ref_inv_sqrt)
    return {"filt": filt, "cov": cov, "feat": feat, "scores": feat @ params.lda_w.T + params.lda_b}


def fit(x: np.ndarray, y: np.ndarray, taps: np.ndarray | None = None,
        shrinkage: float = COV_SHRINKAGE, class_names: tuple[str, ...] | None = None) -> RiemannianLDAParams:
    """Offline CPU training on windows ``x (N, T, C)``, labels ``y (N,)`` (any number of classes)."""
    taps = design_bandpass_taps() if taps is None else taps
    cov = covariance_cpu(bandpass_cpu(x, taps), shrinkage)
    ref = riemann_mean(cov)
    ref_inv_sqrt = _sym_fn(ref, lambda w: w ** -0.5)
    feat = tangent_cpu(cov, ref_inv_sqrt)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(feat, y)
    w, b = lda.coef_, lda.intercept_
    if w.shape[0] == 1:                       # binary: expand to two symmetric score rows
        w, b = np.vstack([-w / 2, w / 2]), np.array([-b[0] / 2, b[0] / 2])
    return RiemannianLDAParams(taps, shrinkage, ref_inv_sqrt, w, b,
                               class_names if class_names is not None else
                               (CLASS_NAMES if w.shape[0] == 2 else tuple(f"Class {i}" for i in range(w.shape[0]))))


@lru_cache(maxsize=2)
def train_default(seed: int = 0, n_per_class: int = 120) -> RiemannianLDAParams:
    """Train on synthetic trials (deterministic for a given seed). Cached per process."""
    from .stream import make_trials
    x, y = make_trials(n_per_class, seed=seed)
    return fit(x, y)


# =========================================================================== torch / GPU
class RiemannianLDA(torch.nn.Module):
    """Batched float64 covariance -> tangent space -> LDA on any torch device."""

    def __init__(self, params: RiemannianLDAParams, device: torch.device | str = "cpu") -> None:
        super().__init__()
        c = params.ref_inv_sqrt.shape[0]
        flat_idx, weights = _triu_spec(c)
        f64 = dict(dtype=torch.float64)
        self.shrinkage = float(params.shrinkage)
        self.n_channels = c
        self.register_buffer("ref_inv_sqrt", torch.from_numpy(params.ref_inv_sqrt).to(**f64))
        self.register_buffer("flat_idx", torch.from_numpy(flat_idx).long())
        self.register_buffer("vec_weights", torch.from_numpy(weights).to(**f64))
        self.register_buffer("eye", torch.eye(c, **f64))
        self.lda = torch.nn.Linear(params.n_features, params.lda_w.shape[0], dtype=torch.float64)
        with torch.no_grad():
            self.lda.weight.copy_(torch.from_numpy(params.lda_w))
            self.lda.bias.copy_(torch.from_numpy(params.lda_b))
        self.to(device).eval()

    def covariance(self, y: torch.Tensor) -> torch.Tensor:
        """``y (B, C, T)`` filtered, mean-removed -> ``(B, C, C)`` float64."""
        y = y.to(torch.float64)
        t = y.shape[-1]
        cov = y @ y.transpose(-1, -2) / (t - 1)
        trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
        cov = cov * (self.n_channels / trace)
        return (1.0 - self.shrinkage) * cov + self.shrinkage * self.eye

    def whiten(self, cov: torch.Tensor) -> torch.Tensor:
        """``(B, C, C)`` SPD -> ``C_ref^{-1/2} C C_ref^{-1/2}`` (symmetrized). Matmuls only: never blocks."""
        m = self.ref_inv_sqrt @ cov @ self.ref_inv_sqrt
        return 0.5 * (m + m.transpose(-1, -2))

    def log_features(self, m: torch.Tensor) -> torch.Tensor:
        """Whitened SPD ``(B, C, C)`` -> ``(B, F)`` tangent-space features via matrix log."""
        evals, evecs = torch.linalg.eigh(m)               # host sync on CUDA (documented)
        logm = (evecs * torch.log(evals).unsqueeze(-2)) @ evecs.transpose(-1, -2)
        return logm.flatten(-2)[:, self.flat_idx] * self.vec_weights

    def tangent(self, cov: torch.Tensor) -> torch.Tensor:
        """``(B, C, C)`` SPD -> ``(B, F)`` tangent-space features."""
        return self.log_features(self.whiten(cov))

    def head(self, feat: torch.Tensor) -> torch.Tensor:
        return self.lda(feat)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.head(self.tangent(self.covariance(y)))


# =========================================================================== ONNX
def export_lda_head_onnx(model: RiemannianLDA, path: str | Path) -> Path:
    """Export the LDA head (``features (B, F) float64 -> scores (B, n_classes)``)."""
    path = Path(path)
    head = copy.deepcopy(model.lda).cpu().eval()      # leave the live (possibly CUDA) model alone
    dummy = torch.zeros(1, head.in_features, dtype=torch.float64)
    # Legacy TorchScript exporter: verified to give a dynamic-batch graph. torch 2.15 marks it
    # deprecated; the torch.export-based exporter needs `onnxscript` and explicit dynamic_shapes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            head, (dummy,), str(path), input_names=["features"], output_names=["scores"],
            dynamic_axes={"features": {0: "batch"}, "scores": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
    return path


class OnnxLDAHead:
    """onnxruntime session for the exported head (CUDA provider if available, else CPU)."""

    def __init__(self, path: str | Path, providers: list[str] | None = None) -> None:
        # torch is imported at module top, i.e. before onnxruntime: per the onnxruntime docs that lets
        # it reuse torch's compatible CUDA/cuDNN libraries (no preload_dlls() needed).
        import onnxruntime as ort
        if providers is None:
            avail = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.providers = self.session.get_providers()

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"features": np.ascontiguousarray(features, dtype=np.float64)})[0]
