import numpy as np

from board import (
    copy_board, get_side, piece_on, encode_move,
    get_pieces, get_white_occ, get_black_occ, get_all_occ,
    get_ep_square, get_castling, set_bit, clear_bit, get_bit,
    pop_lsb, lsb,
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ,
)

# ---- Network hyperparameters (from your training config) ----
HIDDEN_SIZE: int = 128
QA: np.int16 = np.int16(255)
QB: np.int16 = np.int16(64)
SCALE: np.int32 = np.int32(400)

# Piece kind indices
PIECE_CHARS = "PNBRQKpnbrqk"
#              0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king
#              colour = idx // 6  (0=white, 1=black)
#              kind   = idx %  6


def screlu(x: np.int16) -> np.int32:
    """
    Square Clipped ReLU - Activation Function.
    Range is 0.0 .. 1.0 (in other words, 0 to QA*QA quantized).
    """
    y = np.int32(x).clip(0, np.int32(QA))
    return y * y

# ---------------------------------------------------------------------------
# Accumulator / Network
# ---------------------------------------------------------------------------

class Accumulator:
    """
    A column of the feature-weights matrix.
    Equivalent to repr(C, align(64)) — alignment is handled by NumPy internally.
    """

    def __init__(self, vals: np.ndarray):
        # vals: [i16; HIDDEN_SIZE]
        assert vals.dtype == np.int16
        assert vals.shape == (HIDDEN_SIZE,)
        self.vals = vals.copy()

    def add_feature(self, feature_idx: int, net: "Network"):
        """Adds a feature column from the network's feature weights."""
        self.vals += net.feature_weights[feature_idx].vals

    def remove_feature(self, feature_idx: int, net: "Network"):
        """Removes a feature column from the network's feature weights."""
        self.vals -= net.feature_weights[feature_idx].vals


class Network:
    """
    Quantised NNUE network.
    Mirrors the repr(C) Rust struct — fields are loaded in declaration order
    from the binary checkpoint file that bullet produces.

    Layout:
        feature_weights : [Accumulator; 768]  →  shape (768, HIDDEN_SIZE) i16
        feature_bias    : Accumulator         →  shape (HIDDEN_SIZE,)      i16
        output_weights  : [i16; 2*HIDDEN_SIZE]→  shape (2*HIDDEN_SIZE,)    i16
        output_bias     : i16                 →  scalar                    i16
    """

    def __init__(
        self,
        feature_weights: np.ndarray,   # (768, HIDDEN_SIZE) int16
        feature_bias: np.ndarray,      # (HIDDEN_SIZE,)     int16
        output_weights: np.ndarray,    # (2*HIDDEN_SIZE,)   int16
        output_bias: np.int16,
    ):
        assert feature_weights.shape == (768, HIDDEN_SIZE)
        assert feature_bias.shape    == (HIDDEN_SIZE,)
        assert output_weights.shape  == (2 * HIDDEN_SIZE,)

        # Wrap each row of feature_weights as an Accumulator
        self.feature_weights: list[Accumulator] = [
            Accumulator(feature_weights[i]) for i in range(768)
        ]
        self.feature_bias   = Accumulator(feature_bias)
        self.output_weights = output_weights.astype(np.int16)
        self.output_bias    = np.int16(output_bias)

    @classmethod
    def load(cls, path: str) -> "Network":
        """
        Load a quantised bullet checkpoint (.bin) directly from disk.
        Mirrors Rust's:
            std::mem::transmute(*include_bytes!("../checkpoints/quantised_2.bin"))
        Because the Rust struct is repr(C), the binary layout is well-defined
        and we can read it field-by-field in declaration order.
        """
        with open(path, "rb") as f:
            data = f.read()

        offset = 0

        # feature_weights: [Accumulator; 768] = 768 * HIDDEN_SIZE i16 values
        count = 768 * HIDDEN_SIZE
        feature_weights = np.frombuffer(data, dtype=np.int16, count=count, offset=offset)
        feature_weights = feature_weights.reshape((768, HIDDEN_SIZE)).copy()
        offset += count * 2  # 2 bytes per i16

        # feature_bias: Accumulator = HIDDEN_SIZE i16 values
        feature_bias = np.frombuffer(data, dtype=np.int16, count=HIDDEN_SIZE, offset=offset).copy()
        offset += HIDDEN_SIZE * 2

        # output_weights: [i16; 2 * HIDDEN_SIZE]
        count = 2 * HIDDEN_SIZE
        output_weights = np.frombuffer(data, dtype=np.int16, count=count, offset=offset).copy()
        offset += count * 2

        # output_bias: i16 (single scalar)
        output_bias = np.frombuffer(data, dtype=np.int16, count=1, offset=offset)[0]

        return cls(feature_weights, feature_bias, output_weights, output_bias)

    def evaluate(self, us: Accumulator, them: Accumulator) -> np.int32:
        """
        Calculates the output of the network from the two perspective accumulators.

        us   = accumulator from the perspective of the side to move
        them = accumulator from the opponent's perspective
        """
        output = np.int32(0)

        # First half of output_weights corresponds to `us`
        for inp, weight in zip(us.vals, self.output_weights[:HIDDEN_SIZE]):
            output += screlu(inp) * np.int32(weight)

        # Second half corresponds to `them`
        for inp, weight in zip(them.vals, self.output_weights[HIDDEN_SIZE:]):
            output += screlu(inp) * np.int32(weight)

        output //= np.int32(QA)
        output  += np.int32(self.output_bias)
        output  *= SCALE
        output //= np.int32(QA) * np.int32(QB)

        return output

    def make_accumulator(self) -> Accumulator:
        """
        Creates a fresh accumulator initialised to the feature bias.
        Equivalent to Accumulator::new(&net) in Rust.
        """
        return Accumulator(self.feature_bias.vals.copy())


# ---------------------------------------------------------------------------
# Accumulator builder  (Chess768 feature layout)
# ---------------------------------------------------------------------------

def build_accumulators(net: Network, pos: np.ndarray) -> tuple[Accumulator, Accumulator]:
    """
    Build both perspective accumulators from scratch for a given board state.

    Args:
        net : Network  — holds feature_weights and make_accumulator()
        pos : np.ndarray (uint64, length 20) — the board state array

    Returns:
        (us, them) — accumulators for side-to-move and the other side
    """
    us   = net.make_accumulator()
    them = net.make_accumulator()

    stm = int(pos[17])   # 0 = white to move, 1 = black to move

    for p in range(12):
        colour = 0 if p < 6 else 1
        kind   = p % 6

        bb = pos[p]      # bitboard for this piece type
        while bb:
            sq, bb = pop_lsb(bb)

            # Absolute (non-perspective) features
            white_feat = colour * 384 + kind * 64 + sq
            black_feat = (1 - colour) * 384 + kind * 64 + (sq ^ 56)

            # Assign to the correct perspective based on side to move
            if stm == 0:   # white to move → us = white perspective
                us_feat, them_feat = white_feat, black_feat
            else:           # black to move → us = black perspective
                us_feat, them_feat = black_feat, white_feat

            us.add_feature(us_feat, net)
            them.add_feature(them_feat, net)

    return us, them