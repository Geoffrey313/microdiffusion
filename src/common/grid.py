"""
grid.py - shared state-grid geometry for the book-state cells.

The reduced book state is discretised on a fixed grid: the signed queue
imbalance I in [-1, 1] is split into N_I uniform bins and the relative spread S
into M_S bins by per-instrument terciles. A cell index packs the two bins into a
single integer, cell = i_bin * M_S + s_bin, and is used by every analysis that
estimates a per-cell diffusion or count. The grid constants and the packing rule
live here once and are imported wherever a cell index is formed.
"""
from __future__ import annotations
import numpy as np

N_I = 10   # uniform imbalance bins on [-1, 1]
M_S = 3    # per-instrument spread terciles


def cell_ids(I, S, s_edges):
    """Pack imbalance I and spread S into a cell index on the N_I x M_S grid.

    I is mapped to a uniform bin on [-1, 1]; S is placed by its train-estimated
    tercile edges s_edges. Returns cell = i_bin * M_S + s_bin.
    """
    ib = np.clip(((I + 1) / 2 * N_I).astype(int), 0, N_I - 1)
    sb = np.clip(np.searchsorted(s_edges, S, side="right"), 0, M_S - 1)
    return ib * M_S + sb
