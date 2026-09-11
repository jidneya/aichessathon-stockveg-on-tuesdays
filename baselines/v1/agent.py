import os
import shutil

# =============================================================================
# CLEAR NUMBA __pycache__ BEFORE ANY NUMBA IMPORTS
# Must happen here — before numba or src.* are imported — so that Numba
# always recompiles fresh and never loads stale zero-weight cached kernels.
# =============================================================================
def _clear_numba_cache():
    base = os.path.dirname(os.path.abspath(__file__))
    targets = [
        os.path.join(base, "__pycache__"),
        os.path.join(base, "src", "__pycache__"),
    ]
    for path in targets:
        if os.path.isdir(path):
            shutil.rmtree(path)

_clear_numba_cache()

# =============================================================================
# NOW safe to import Numba-compiled modules
# =============================================================================
import numpy as np
from numba import njit, uint64
from numba.typed import Dict
from numba.core import types

from src.board import board_from_fen, decode_move
from src.search import get_best_move, negamax, create_tt
from src.evaluate import initialize_nnue

# =============================================================================
# NNUE INITIALISATION  (runs once per process)
# =============================================================================
#print("Initializing NNUE evaluation...", flush=True)
initialize_nnue("baselines/v1/src/checkpoints/quantised.bin")
#print("NNUE ready!", flush=True)


def get_move(fen: str, time_left_ms: int) -> str:
    """
    Called by the harness for every move.
    Returns a UCI move string e.g. 'e2e4', 'e7e8q'.
    """
    #print(f"Searching position: {fen[:50]}...", flush=True)
    #print(f"Time remaining: {time_left_ms}ms", flush=True)

    board_array = board_from_fen(fen)
    best_encoded_move = get_best_move(board_array, time_left_ms)

    from_sq = best_encoded_move & 63
    to_sq   = (best_encoded_move >> 6) & 63
    promo   = (best_encoded_move >> 12) & 15

    files = 'abcdefgh'
    def sq_to_uci(sq):
        return files[sq % 8] + str(sq // 8 + 1)

    uci = sq_to_uci(from_sq) + sq_to_uci(to_sq)

    promo_map = {0: '', 1: 'n', 2: 'b', 3: 'r', 4: 'q',
                 9: 'n', 10: 'b', 11: 'r', 12: 'q'}
    if promo in promo_map and promo_map[promo]:
        uci += promo_map[promo]

    #print(f"Best move: {uci}", flush=True)
    return uci
