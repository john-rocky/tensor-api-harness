#!/usr/bin/env python3
"""Writes the block_match fixtures next to this file (deterministic; rerun after changing them).

ref/        block_00.f32 [2, 8], logits.npy [2, 6]
got_ok/     the reference plus noise well inside the bar
got_bad/    block_00 with one element moved by 0.5 (fails max_abs)
got_short/  block_00 with one element missing (element-count mismatch)
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def write(sub, block, logits):
    d = os.path.join(HERE, sub)
    os.makedirs(d, exist_ok=True)
    block.astype("<f4").tofile(os.path.join(d, "block_00.f32"))
    np.save(os.path.join(d, "logits.npy"), logits.astype(np.float32))


def main():
    rng = np.random.RandomState(0)
    block = rng.randn(2, 8).astype(np.float32)
    logits = (rng.randn(2, 6) * 4).astype(np.float32)
    write("ref", block, logits)
    write("got_ok", block + rng.randn(2, 8).astype(np.float32) * 1e-5,
          logits + rng.randn(2, 6).astype(np.float32) * 1e-4)
    bad = block.copy()
    bad[1, 3] += 0.5
    write("got_bad", bad, logits)
    write("got_short", block.ravel()[:-1], logits)


if __name__ == "__main__":
    main()
