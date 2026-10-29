"""Shared fixtures. GPU-only tests skip without CUDA, unless BCI_REQUIRE_CUDA=1 (then they fail)."""

from __future__ import annotations

import os

import pytest
import torch

from bci.classifier import train_default

REQUIRE_CUDA = os.environ.get("BCI_REQUIRE_CUDA", "0") == "1"


def require_cuda() -> None:
    if torch.cuda.is_available():
        return
    msg = f"CUDA is not available (torch {torch.__version__}, built for CUDA {torch.version.cuda})"
    if REQUIRE_CUDA:
        pytest.fail(f"BCI_REQUIRE_CUDA=1 but {msg}. On Windows the default PyPI torch wheel is "
                    "CPU-only; install a CUDA build (see README).", pytrace=False)
    pytest.skip(msg)


@pytest.fixture(scope="session")
def params():
    """Classifier trained once on synthetic trials (deterministic)."""
    return train_default()


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request) -> str:
    if request.param == "cuda":
        require_cuda()
    return request.param
