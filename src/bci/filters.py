"""Band-pass filter: scipy CPU reference and a vectorized torch FIR (optionally torch.compile'd).

The GPU filter is one ``conv1d`` over all channels at once (channels folded into the
batch axis) — there is no Python loop over the 64 channels. It computes a "valid"
convolution, so a 500-sample window yields 500 - 101 + 1 = 400 samples, then removes each
channel's mean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import firwin, lfilter

from . import BAND_HZ, FIR_TAPS, FS_HZ, N_CHANNELS, WINDOW_SAMPLES

log = logging.getLogger(__name__)


def configure_precision() -> None:
    """Disable TF32 everywhere. PyTorch leaves ``cudnn.allow_tf32`` ON by default, which
    would round the ``conv1d`` inputs to 10-bit mantissas and break the atol=1e-4 contract."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def design_bandpass_taps(numtaps: int = FIR_TAPS, band: tuple[float, float] = BAND_HZ,
                         fs: float = FS_HZ) -> np.ndarray:
    """Hamming-window FIR band-pass taps, float64, shape ``(numtaps,)``."""
    return firwin(numtaps, list(band), pass_zero=False, fs=fs, window="hamming")


def bandpass_cpu(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """CPU reference (scipy, float64). ``x (B, T, C)`` or ``(T, C)`` -> ``(B, C, T-K+1)``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    y = lfilter(taps, [1.0], x, axis=1)[:, len(taps) - 1:, :]      # drop the start-up transient
    y = np.transpose(y, (0, 2, 1))
    return y - y.mean(axis=-1, keepdims=True)


class FIRBandpass(torch.nn.Module):
    """``(B, T, C)`` float32 -> ``(B, C, T-K+1)`` float32, mean-removed per channel."""

    def __init__(self, taps: np.ndarray) -> None:
        super().__init__()
        # conv1d is cross-correlation; flip the taps so it equals scipy's convolution.
        kernel = torch.from_numpy(np.ascontiguousarray(taps[::-1])).to(torch.float32)
        self.register_buffer("kernel", kernel.view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        xc = x.permute(0, 2, 1).reshape(b * c, 1, t)       # channels folded into the batch axis
        y = F.conv1d(xc, self.kernel).view(b, c, -1)
        return y - y.mean(dim=-1, keepdim=True)


@dataclass(frozen=True)
class FilterInfo:
    compiled: bool
    note: str


def make_bandpass(taps: np.ndarray, device: torch.device | str, compile: bool | str = "auto",
                  example_shape: tuple[int, int, int] = (1, WINDOW_SAMPLES, N_CHANNELS),
                  ) -> tuple[Callable[[torch.Tensor], torch.Tensor], FilterInfo]:
    """Build the filter on ``device``.

    ``compile``: ``True`` = try ``torch.compile``; ``False`` = eager; ``"auto"`` = compile on
    CUDA only. If compilation fails (Inductor needs Triton for CUDA and a C++ compiler for
    CPU) this **falls back to eager** with a warning and ``FilterInfo.compiled == False``,
    so result files never claim a compiled filter that did not run.
    """
    configure_precision()
    device = torch.device(device)
    module = FIRBandpass(taps).to(device).eval()
    want = (device.type == "cuda") if compile == "auto" else bool(compile)
    if not want:
        return module, FilterInfo(False, "eager (compile not requested)")
    try:
        compiled = torch.compile(module, fullgraph=True, dynamic=False)
        with torch.no_grad():                       # trigger compilation now, not on the first chunk
            compiled(torch.zeros(example_shape, dtype=torch.float32, device=device))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return compiled, FilterInfo(True, "torch.compile")
    except Exception as exc:  # noqa: BLE001 - any backend failure means "use eager"
        log.warning("torch.compile unavailable (%s: %s); using eager filter",
                    type(exc).__name__, str(exc).splitlines()[0][:200])
        return module, FilterInfo(False, f"eager (compile failed: {type(exc).__name__})")
