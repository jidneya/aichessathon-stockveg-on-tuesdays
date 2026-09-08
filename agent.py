import numpy as np

# =============================================================================
# STEP 1 — Load NNUE weights FIRST, before search.py is imported.
#
# search.py calls _warmup() at module level (import time).  _warmup() runs
# a depth-1 search with zero weights so Numba JIT-compiles everything during
# the 90s init budget.  The real weights are passed as explicit arguments at
# search time, so the Numba cache (which stores only machine code, never data)
# is always valid — no cache clearing needed between games.
# =============================================================================
from src.evaluate import initialize_nnue, L0_WEIGHTS, L0_BIASES, L1_WEIGHTS, L1_BIAS

print("Initializing NNUE evaluation...", flush=True)
initialize_nnue("src/checkpoints/1_simple/quantised.bin")
print("NNUE ready!", flush=True)

# =============================================================================
# STEP 2 — Import search AFTER weights are loaded.
#
# _warmup() fires here.  It uses zero weights for compilation (safe, because
# weights are arguments not globals in every @njit kernel), so the resulting
# cache is correct for all subsequent calls with real weights.
# =============================================================================
from src.board import board_from_fen
from src.search import get_best_move


def get_move(fen: str, time_left_ms: int) -> str:
    """
    Called by the harness for every move.
    Returns a UCI move string e.g. 'e2e4', 'e7e8q'.
    """
    print(f"Searching position: {fen[:50]}...", flush=True)
    print(f"Time remaining: {time_left_ms}ms", flush=True)

    board_array = board_from_fen(fen)

    # Pass the already-loaded weight arrays explicitly so every @njit kernel
    # receives real weights at call-time (never baked into the cache).
    best_encoded_move = get_best_move(
        board_array, time_left_ms,
        l0w=L0_WEIGHTS, l0b=L0_BIASES, l1w=L1_WEIGHTS, l1_bias=L1_BIAS,
    )

    from_sq = best_encoded_move & 63
    to_sq   = (best_encoded_move >> 6) & 63
    promo   = (best_encoded_move >> 12) & 15

    files = 'abcdefgh'
    def sq_to_uci(sq):
        return files[sq % 8] + str(sq // 8 + 1)

    uci = sq_to_uci(from_sq) + sq_to_uci(to_sq)

    promo_map = {0: '', 1: 'n', 2: 'b', 3: 'r', 4: 'q',
                 7: 'n', 8: 'b', 9: 'r', 10: 'q'}
    if promo in promo_map and promo_map[promo]:
        uci += promo_map[promo]

    print(f"Best move: {uci}", flush=True)
    return uci
