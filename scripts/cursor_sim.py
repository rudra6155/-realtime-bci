"""Replay the decisions of the session-E stream through the cursor and measure how often it sits at an edge.

    python scripts/cursor_sim.py --decoder fbcsp --out results/real/cursor_sim_fbcsp.csv

For each subject the decoder is trained on session T, session E is streamed through the pipeline (as in
``real_benchmark accuracy``), and the per-chunk decisions drive a cursor continuously over the whole replay (the
dashboard does the same). Two control laws are compared: the plain integrator (``decay = 1``) and the leaky one the
dashboard uses (``decay = 0.95``). Both were fixed before this was run; nothing is tuned here.

Columns: ``pinned`` = share of decisions with |x| >= 0.999; ``sign_ok`` = share of left/right-imagery chunks on which the
cursor is on the intended side (x < 0 for left, x > 0 for right; x = 0 counts as wrong; chance is about 0.5).
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from bci import real_benchmark as rb
from bci.closed_loop import DEFAULT_CURSOR_DECAY, DEFAULT_SMOOTH_K, CursorController
from bci.stream import RealEEGStream, load_bciciv2a


def simulate(subject: int, decoder: str, smooth_k: int) -> list[dict]:
    params = rb.train_subject(subject, decoder=decoder)
    rec = load_bciciv2a(subject, "1test")
    preds = rb.stream_session(params, rec, "cpu", smooth_k=smooth_k)
    seq = []                                                   # (decision, intent) over the whole replay, in order
    for r in range(len(rec.runs)):
        ends, scores = preds[r]
        labels = {(c.index + 1) * 64: c.label for c in RealEEGStream(recording=rec, runs=[r])}
        seq += [(int(np.argmax(sc)), labels[int(e)]) for e, sc in zip(ends, scores)]
    rows = []
    for decay in (1.0, DEFAULT_CURSOR_DECAY):
        c = CursorController(params.class_names, gain=0.1, decay=decay)
        x = np.array([c.update(d, 0).x for d, _ in seq])
        lab = np.array([b for _, b in seq])
        lr = (lab == 0) | (lab == 1)
        ok = np.where(lab == 0, x < 0, x > 0)
        rows.append({"subject": subject, "decoder": decoder, "decay": decay, "decisions": len(x),
                     "pinned": float((np.abs(x) >= 0.999).mean()), "left_right_chunks": int(lr.sum()),
                     "sign_ok": float(ok[lr].mean()), "mean_abs_x": float(np.abs(x).mean())})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--decoder", choices=rb.DECODERS, default=rb.DEFAULT_DECODER)
    p.add_argument("--smooth-k", type=int, default=DEFAULT_SMOOTH_K["real"])
    p.add_argument("--subjects", default="1-9")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    rows = []
    for s in rb.parse_subjects(args.subjects):
        rows += simulate(s, args.decoder, args.smooth_k)
        print(f"S{s} done", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    for decay in sorted({r["decay"] for r in rows}, reverse=True):
        sel = [r for r in rows if r["decay"] == decay]
        print(f"decay {decay}: pinned {np.mean([r['pinned'] for r in sel]):.3f}, sign_ok {np.mean([r['sign_ok'] for r in sel]):.3f}, "
              f"mean |x| {np.mean([r['mean_abs_x'] for r in sel]):.3f} (mean over {len(sel)} subjects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
