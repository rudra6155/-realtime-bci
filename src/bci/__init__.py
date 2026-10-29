"""realtime-bci: a streaming, GPU-accelerated EEG decoding pipeline on synthetic data.

The constants here are the single source of truth for the shapes in SPEC.md.
"""

__version__ = "0.1.0"

FS_HZ = 250                      # sampling rate
N_CHANNELS = 64                  # EEG channels
CHUNK_SAMPLES = 64               # samples per chunk (256 ms at 250 Hz)
RING_CAPACITY = 1000             # ring buffer length in samples (4 s)
WINDOW_SAMPLES = 500             # decode window read from the ring buffer (2 s)

BAND_HZ = (8.0, 30.0)            # band-pass edges
FIR_TAPS = 101                   # FIR length -> 400 valid output samples per window
COV_SHRINKAGE = 0.05             # covariance shrinkage toward identity

CLASS_NAMES = ("Left Hand", "Right Hand")
LATENCY_TARGET_MS = 50.0         # p99 target, chunk received -> label ready

CHUNK_PERIOD_S = CHUNK_SAMPLES / FS_HZ
