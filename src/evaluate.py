import os
import numpy as np
from numba import njit, int32, uint64
import struct

# =============================================================================
# NETWORK ARCHITECTURE
# =============================================================================
H  = 128   # hidden layer size
IN = 768   # input features (12 pieces × 64 squares)

# =============================================================================
# QUANTISATION CONSTANTS
# =============================================================================
QA    = np.int32(255)
QB    = np.int32(64)
SCALE = np.int32(400)

# =============================================================================
# WEIGHT ARRAYS  (populated by initialize_nnue)
# =============================================================================
L0_WEIGHTS: np.ndarray | None = None   # shape (H, IN)  int32
L0_BIASES:  np.ndarray | None = None   # shape (H,)     int32
L1_WEIGHTS: np.ndarray | None = None   # shape (2*H,)   int32
L1_BIAS:    np.int32   | None = None   # scalar         int32

# =============================================================================
# PIECE-SQUARE TABLES
# =============================================================================
# Layout: PST_TABLE[row, square]
#   Rows 0-5  : White  Pawn/Knight/Bishop/Rook/Queen/King-MG
#   Rows 6-11 : Black  (vertical mirror of rows 0-5)
#   Row  12   : White King-EG
#   Row  13   : Black King-EG
#
# Square encoding: 0=a1 … 63=h8  (rank 1 at index 0)
# Tables are written rank-by-rank from rank 1 (bottom) to rank 8 (top)
# so that index 0 = a1, index 8 = a2, etc.
# =============================================================================

def _mirror(table: list) -> list:
    """Flip a 64-entry table vertically (rank 1 ↔ rank 8)."""
    out = []
    for rank in range(7, -1, -1):
        out.extend(table[rank * 8: rank * 8 + 8])
    return out

# ── Pawn (White, rank 1 at bottom) ───────────────────────────────────────────
_PAWN_W = [
     0,  0,  0,  0,  0,  0,  0,  0,   # rank 1 (never occupied)
    -5, -5, -5, -5, -5, -5, -5, -5,   # rank 2  (discourage staying back)
    -5, -5,  0,  5,  5,  0, -5, -5,   # rank 3
     0,  0,  5, 20, 20,  5,  0,  0,   # rank 4  (centre bonus)
     5,  5, 10, 25, 25, 10,  5,  5,   # rank 5
    10, 10, 20, 30, 30, 20, 10, 10,   # rank 6
    50, 50, 50, 50, 50, 50, 50, 50,   # rank 7  (near promotion)
     0,  0,  0,  0,  0,  0,  0,  0,   # rank 8
]

# ── Knight ────────────────────────────────────────────────────────────────────
_KNIGHT_W = [
   -50,-40,-30,-30,-30,-30,-40,-50,   # rank 1
   -40,-20,  0,  0,  0,  0,-20,-40,   # rank 2
   -30,  0, 10, 15, 15, 10,  0,-30,   # rank 3
   -30,  5, 15, 20, 20, 15,  5,-30,   # rank 4
   -30,  0, 15, 20, 20, 15,  0,-30,   # rank 5
   -30,  5, 10, 15, 15, 10,  5,-30,   # rank 6
   -40,-20,  0,  5,  5,  0,-20,-40,   # rank 7
   -50,-40,-30,-30,-30,-30,-40,-50,   # rank 8
]

# ── Bishop ────────────────────────────────────────────────────────────────────
_BISHOP_W = [
   -20,-10,-10,-10,-10,-10,-10,-20,   # rank 1
   -10,  0,  0,  0,  0,  0,  0,-10,   # rank 2
   -10,  0,  5, 10, 10,  5,  0,-10,   # rank 3
   -10,  5,  5, 10, 10,  5,  5,-10,   # rank 4
   -10,  0, 10, 10, 10, 10,  0,-10,   # rank 5
   -10, 10, 10, 10, 10, 10, 10,-10,   # rank 6
   -10,  5,  0,  0,  0,  0,  5,-10,   # rank 7
   -20,-10,-10,-10,-10,-10,-10,-20,   # rank 8
]

# ── Rook ──────────────────────────────────────────────────────────────────────
_ROOK_W = [
     0,  0,  0,  5,  5,  0,  0,  0,   # rank 1
    -5,  0,  0,  0,  0,  0,  0, -5,   # rank 2
    -5,  0,  0,  0,  0,  0,  0, -5,   # rank 3
    -5,  0,  0,  0,  0,  0,  0, -5,   # rank 4
    -5,  0,  0,  0,  0,  0,  0, -5,   # rank 5
    -5,  0,  0,  0,  0,  0,  0, -5,   # rank 6
     5, 10, 10, 10, 10, 10, 10,  5,   # rank 7
     0,  0,  0,  0,  0,  0,  0,  0,   # rank 8
]

# ── Queen ─────────────────────────────────────────────────────────────────────
_QUEEN_W = [
   -20,-10,-10, -5, -5,-10,-10,-20,   # rank 1
   -10,  0,  0,  0,  0,  0,  0,-10,   # rank 2
   -10,  0,  5,  5,  5,  5,  0,-10,   # rank 3
    -5,  0,  5,  5,  5,  5,  0, -5,   # rank 4
     0,  0,  5,  5,  5,  5,  0, -5,   # rank 5
   -10,  5,  5,  5,  5,  5,  0,-10,   # rank 6
   -10,  0,  5,  0,  0,  0,  0,-10,   # rank 7
   -20,-10,-10, -5, -5,-10,-10,-20,   # rank 8
]

# ── King — middlegame ─────────────────────────────────────────────────────────
_KING_MG_W = [
    20, 30, 20, 20, 20, 20, 30, 20,   # rank 1 (flattened centre)
    20, 20,  0,  0,  0,  0, 20, 20,   # rank 2 (discourage stepping up)
   -10,-20,-20,-20,-20,-20,-20,-10,   # rank 3
   -20,-30,-30,-40,-40,-30,-30,-20,   # rank 4
   -30,-40,-40,-50,-50,-40,-40,-30,   # rank 5
   -30,-40,-40,-50,-50,-40,-40,-30,   # rank 6
   -30,-40,-40,-50,-50,-40,-40,-30,   # rank 7
   -30,-40,-40,-50,-50,-40,-40,-30,   # rank 8
]

# ── King — endgame ────────────────────────────────────────────────────────────
_KING_EG_W = [
   -50,-40,-30,-20,-20,-30,-40,-50,   # rank 1
   -30,-20,-10,  0,  0,-10,-20,-30,   # rank 2
   -30,-10, 20, 30, 30, 20,-10,-30,   # rank 3
   -30,-10, 30, 40, 40, 30,-10,-30,   # rank 4
   -30,-10, 30, 40, 40, 30,-10,-30,   # rank 5
   -30,-10, 20, 30, 30, 20,-10,-30,   # rank 6
   -30,-30,  0,  0,  0,  0,-30,-30,   # rank 7
   -50,-30,-30,-30,-30,-30,-30,-50,   # rank 8
]


def _build_pst() -> np.ndarray:
    """
    Returns shape (14, 64) int32 array.
    Rows 0-5  : White  Pawn/Knight/Bishop/Rook/Queen/King-MG
    Rows 6-11 : Black  (vertical mirror of White rows 0-5)
    Row  12   : White King-EG
    Row  13   : Black King-EG
    """
    tables_w = [_PAWN_W, _KNIGHT_W, _BISHOP_W, _ROOK_W, _QUEEN_W, _KING_MG_W]
    pst = np.zeros((14, 64), dtype=np.int32)
    for i, t in enumerate(tables_w):
        pst[i]     = np.array(t,          dtype=np.int32)
        pst[i + 6] = np.array(_mirror(t), dtype=np.int32)
    pst[12] = np.array(_KING_EG_W,          dtype=np.int32)
    pst[13] = np.array(_mirror(_KING_EG_W), dtype=np.int32)
    return pst


PST_TABLE: np.ndarray = _build_pst()

# =============================================================================
# NNUE LOADER  —  loads raw.bin (no-header float32 dump)
# =============================================================================
CHECKPOINT = os.path.join(
    os.path.dirname(__file__),
    "checkpoints", "1_simple", "raw.bin"
)


def initialize_nnue() -> None:
    global L0_WEIGHTS, L0_BIASES, L1_WEIGHTS, L1_BIAS
    print("Initializing NNUE evaluation...")

    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"NNUE checkpoint not found: {CHECKPOINT}\n"
            f"Expected raw.bin (no-header float32 dump) in checkpoints/1_simple/"
        )

    # ── Load raw float32 weights ──────────────────────────────────────────────
    data = np.fromfile(CHECKPOINT, dtype=np.float32)

    expected = H * IN + H + 2 * H + 1
    if len(data) != expected:
        raise ValueError(
            f"raw.bin has {len(data)} float32 values, expected {expected} "
            f"(H={H}, IN={IN}). Check network architecture constants."
        )

    idx = 0

    # L0 weights: (H, IN) — quantise to int32 with scale QA
    w0_f = data[idx: idx + H * IN].reshape(H, IN);  idx += H * IN
    L0_WEIGHTS = np.round(w0_f * float(QA)).astype(np.int32)

    # L0 biases: (H,) — quantise with scale QA
    b0_f = data[idx: idx + H];  idx += H
    L0_BIASES = np.round(b0_f * float(QA)).astype(np.int32)

    # L1 weights: (2H,) — quantise with scale QB
    w1_f = data[idx: idx + 2 * H];  idx += 2 * H
    L1_WEIGHTS = np.round(w1_f * float(QB)).astype(np.int32)

    # L1 bias: scalar — quantise with scale QA*QB
    b1_f = data[idx]
    L1_BIAS = np.int32(round(float(b1_f) * float(QA) * float(QB)))

    print(f" Loaded NNUE checkpoint from {CHECKPOINT}")
    print(f"  L0 weights: {L0_WEIGHTS.shape}, L0 biases: {L0_BIASES.shape}")
    print(f"  L1 weights: {L1_WEIGHTS.shape}, L1 bias: scalar")
    print("NNUE ready!")


# =============================================================================
# JIT KERNELS
# =============================================================================

@njit(cache=True)
def _screlu(x: int32, qa: int32) -> int32:
    """Squared Clipped ReLU: clamp(x, 0, QA)^2"""
    if x < int32(0):
        x = int32(0)
    if x > qa:
        x = qa
    return x * x


@njit(cache=True)
def nnue_forward(
    board,
    w0: np.ndarray,   # (H, IN)  int32
    b0: np.ndarray,   # (H,)     int32
    w1: np.ndarray,   # (2H,)    int32
    b1: int32,
    qa: int32,
    qb: int32,
    scale: int32,
) -> int32:
    """
    Two-perspective NNUE forward pass.
    Returns centipawn score from the side-to-move's perspective.
    """
    side = int(board[17])   # 0=White, 1=Black
    H_  = w0.shape[0]
    IN_ = w0.shape[1]

    # ── Accumulate both perspectives ──────────────────────────────────────────
    acc_stm  = b0.copy()   # side-to-move accumulator
    acc_ntm  = b0.copy()   # non-side-to-move accumulator

    for piece in range(12):
        bb = board[piece]
        while bb:
            # pop LSB
            lsb_bb = bb & (~bb + uint64(1))
            sq = int32(0)
            tmp = lsb_bb
            while tmp > uint64(1):
                tmp >>= uint64(1)
                sq += int32(1)
            bb ^= lsb_bb

            # White perspective: piece*64 + sq
            # Black perspective: piece*64 + (sq ^ 56)  [vertical flip]
            if side == 0:
                feat_stm = piece * IN_ // 12 + int(sq)
                feat_ntm = piece * IN_ // 12 + (int(sq) ^ 56)
            else:
                feat_stm = piece * IN_ // 12 + (int(sq) ^ 56)
                feat_ntm = piece * IN_ // 12 + int(sq)

            if 0 <= feat_stm < IN_:
                for i in range(H_):
                    acc_stm[i] += w0[i, feat_stm]
            if 0 <= feat_ntm < IN_:
                for i in range(H_):
                    acc_ntm[i] += w0[i, feat_ntm]

    # ── Output layer ──────────────────────────────────────────────────────────
    out = b1
    for i in range(H_):
        out += _screlu(acc_stm[i], qa) * w1[i]
        out += _screlu(acc_ntm[i], qa) * w1[H_ + i]

    return int32((int(out) * int(scale)) // (int(qa) * int(qa) * int(qb)))


@njit(cache=True)
def pst_score(board, pst: np.ndarray) -> int32:
    """
    Compute PST score + Material Base Value from side-to-move's perspective.
    """
    # ── Material values for endgame detection (pawns excluded) ────────────────
    MAT = (int32(0), int32(320), int32(330), int32(500), int32(900), int32(0),
           int32(0), int32(320), int32(330), int32(500), int32(900), int32(0))

    # ── Base material values for evaluation (pawns included) ──────────────────
    BASE_VALS = (int32(100), int32(320), int32(330), int32(500), int32(900), int32(0),
                 int32(100), int32(320), int32(330), int32(500), int32(900), int32(0))

    total_mat = int32(0)
    for p in range(12):
        if p == 0 or p == 6:   # skip pawns
            continue
        bb = board[p]
        cnt = int32(0)
        while bb:
            bb &= (bb - uint64(1))
            cnt += int32(1)
        total_mat += MAT[p] * cnt

    is_endgame = total_mat <= int32(1300)

    side = int(board[17])   # 0=White, 1=Black
    score = int32(0)

    for piece in range(12):
        bb = board[piece]
        while bb:
            # pop LSB safely
            lsb_bb = bb & (~bb + uint64(1))
            sq = int32(0)
            tmp = lsb_bb
            while tmp > uint64(1):
                tmp >>= uint64(1)
                sq += int32(1)
            bb ^= lsb_bb

            sq_i = int(sq)

            # Determine PST row
            if piece == 5:   # White King
                row = 12 if is_endgame else 5
            elif piece == 11:  # Black King
                row = 13 if is_endgame else 11
            else:
                row = piece   # rows 0-4 for White, 6-10 for Black

            # Add BOTH the base material value and the positional PST value
            val = pst[row, sq_i] + BASE_VALS[piece]

            # White pieces add positively, Black pieces subtract
            if piece < 6:
                score += val
            else:
                score -= val

    # Flip to side-to-move perspective
    if side == 1:
        score = -score

    return score

@njit(cache=True)
def evaluate(
    board,
    w0: np.ndarray,
    b0: np.ndarray,
    w1: np.ndarray,
    b1: int32,
    qa: int32,
    qb: int32,
    scale: int32,
    pst: np.ndarray,
) -> int32:
    """Combined NNUE + PST evaluation from side-to-move's perspective."""
    nnue = nnue_forward(board, w0, b0, w1, b1, qa, qb, scale)
    pst_val = pst_score(board, pst)
    return nnue + pst_val
