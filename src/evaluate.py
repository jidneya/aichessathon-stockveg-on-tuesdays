import numpy as np
from numba import njit, int16, int32

from src.board import piece_on, get_side

# =============================================================================
# NNUE ARCHITECTURE CONSTANTS
# =============================================================================
INPUT_SIZE  = 768   # 12 piece types × 64 squares
HIDDEN_SIZE = 128

QA  = 255
QB  = 64
SCALE = 400

# =============================================================================
# GLOBAL WEIGHT ARRAYS  (populated by initialize_nnue, passed into @njit)
# =============================================================================
# Initialised to zeros; replaced with real values by initialize_nnue().
# IMPORTANT: these are passed as *arguments* into every @njit kernel so that
# Numba's cache stores only compiled code, never the weight data.  This means
# the cache is always valid across runs — no need to delete __pycache__.
L0_WEIGHTS = np.zeros((HIDDEN_SIZE, INPUT_SIZE), dtype=np.int16)
L0_BIASES  = np.zeros(HIDDEN_SIZE,               dtype=np.int16)
L1_WEIGHTS = np.zeros(2 * HIDDEN_SIZE,           dtype=np.int16)
L1_BIAS    = np.int32(0)

# =============================================================================
# LOAD CHECKPOINT
# =============================================================================
def initialize_nnue(checkpoint_path: str = "checkpoints/1_simple/quantised.bin"):
    """
    Load quantised NNUE weights from a bullet checkpoint.

    File layout (little-endian int16):
      1. L0 weights : HIDDEN_SIZE × INPUT_SIZE  (column-major / Fortran order)
      2. L0 biases  : HIDDEN_SIZE
      3. L1 weights : 2 × HIDDEN_SIZE
      4. L1 bias    : 1 scalar
    """
    global L0_WEIGHTS, L0_BIASES, L1_WEIGHTS, L1_BIAS

    with open(checkpoint_path, "rb") as f:
        l0w_flat   = np.fromfile(f, dtype=np.int16, count=HIDDEN_SIZE * INPUT_SIZE)
        L0_WEIGHTS = l0w_flat.reshape((HIDDEN_SIZE, INPUT_SIZE), order="F")
        L0_BIASES  = np.fromfile(f, dtype=np.int16, count=HIDDEN_SIZE)
        L1_WEIGHTS = np.fromfile(f, dtype=np.int16, count=2 * HIDDEN_SIZE)
        L1_BIAS    = np.int32(np.fromfile(f, dtype=np.int16, count=1)[0])

    print(f" Loaded NNUE checkpoint from {checkpoint_path}")
    print(f"  L0 weights: {L0_WEIGHTS.shape}, L0 biases: {L0_BIASES.shape}")
    print(f"  L1 weights: {L1_WEIGHTS.shape}, L1 bias: scalar")

# =============================================================================
# FEATURE EXTRACTION  (no globals captured — cache-safe)
# =============================================================================
@njit(cache=True)
def get_feature_index(piece: int, square: int, perspective: int) -> int:
    """Chess-768 feature index for a single piece."""
    if perspective == 1:
        square = square ^ 56          # rank-flip for black's view
    if perspective == 1:
        piece_idx = piece - 6 if piece >= 6 else piece + 6
    else:
        piece_idx = piece
    return piece_idx * 64 + square


@njit(cache=True)
def extract_active_features(board):
    """
    Return (stm_features, ntm_features) as int32 arrays.
    No global weight arrays are touched here — fully cache-safe.
    """
    side = get_side(board)
    stm_list = np.zeros(32, dtype=np.int32)
    ntm_list = np.zeros(32, dtype=np.int32)
    n = 0
    for sq in range(64):
        p = piece_on(board, sq)
        if p == -1:
            continue
        if side == 0:
            stm_list[n] = get_feature_index(p, sq, 0)
            ntm_list[n] = get_feature_index(p, sq, 1)
        else:
            stm_list[n] = get_feature_index(p, sq, 1)
            ntm_list[n] = get_feature_index(p, sq, 0)
        n += 1
    return stm_list[:n], ntm_list[:n]

# =============================================================================
# NNUE FORWARD PASS  (weights passed as arguments — never baked into cache)
# =============================================================================
@njit(cache=True)
def _screlu(x: int) -> int:
    c = x if x > 0 else 0
    c = c if c < QA else QA
    return c * c


@njit(cache=True)
def _forward_l0(features, l0w, l0b):
    """Sparse accumulation + SCReLU for one perspective."""
    acc = l0b.copy().astype(np.int32)
    for feat in features:
        for i in range(HIDDEN_SIZE):
            acc[i] += int32(l0w[i, feat])
    out = np.zeros(HIDDEN_SIZE, dtype=np.int32)
    for i in range(HIDDEN_SIZE):
        out[i] = _screlu(acc[i])
    return out


@njit(cache=True)
def _forward_l1(stm_h, ntm_h, l1w, l1_bias):
    """Output layer: dot-product, dequantise, scale."""
    total = int32(0)
    for i in range(HIDDEN_SIZE):
        total += stm_h[i] * int32(l1w[i])
    for i in range(HIDDEN_SIZE):
        total += ntm_h[i] * int32(l1w[HIDDEN_SIZE + i])
    total = total // QA
    total += l1_bias
    total  = total * SCALE
    total  = total // (QA * QB)
    return int(total)


@njit(cache=True)
def nnue_forward(board, l0w, l0b, l1w, l1_bias):
    """
    Full NNUE forward pass — callable from @njit (negamax).
    All weight arrays are explicit arguments — Numba's cache stores only the
    compiled machine code, never the weight values.
    """
    stm_feat, ntm_feat = extract_active_features(board)
    stm_h = _forward_l0(stm_feat, l0w, l0b)
    ntm_h = _forward_l0(ntm_feat, l0w, l0b)
    return _forward_l1(stm_h, ntm_h, l1w, l1_bias)


# =============================================================================
# PUBLIC API  — called from search.py
# =============================================================================
def evaluate(board) -> int:
    """
    Evaluate *board* with the NNUE network.
    Returns centipawns from the side-to-move's perspective.
    The global weight arrays are injected here at call-time so the @njit
    kernel never captures them as compile-time constants.

    NOTE: This is a plain Python function — do NOT call it from inside
    a @njit function.  Call nnue_forward(...) directly instead, passing
    the weight arrays as arguments (see search.py).
    """
    return nnue_forward(board, L0_WEIGHTS, L0_BIASES, L1_WEIGHTS, L1_BIAS)
