import numpy as np

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
# Piece / Position
# ---------------------------------------------------------------------------

class Piece:
    """
    One piece on the board, in absolute (non-perspective-flipped) terms.

    colour : 0 = white, 1 = black
    kind   : 0=pawn 1=knight 2=bishop 3=rook 4=queen 5=king
    square : 0..63  (a1=0, h8=63)
    """

    def __init__(self, colour: int, kind: int, square: int):
        self.colour = colour
        self.kind   = kind
        self.square = square


class Position:
    """
    A board position: a list of pieces and the side to move.

    stm : 0 = white to move, 1 = black to move
    """

    def __init__(self, pieces: list[Piece], stm: int):
        self.pieces = pieces
        self.stm    = stm


# ---------------------------------------------------------------------------
# FEN parser
# ---------------------------------------------------------------------------

def parse_fen(fen: str) -> Position:
    """
    Parses a FEN string into a Position.
    Mirrors the Rust parse_fen function exactly.
    Raises ValueError on malformed input.
    """
    parts = fen.split()
    if not parts:
        raise ValueError("empty FEN")

    board_str = parts[0]

    # Side to move
    stm = 1 if (len(parts) > 1 and parts[1] == "b") else 0

    ranks = board_str.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN board must have 8 ranks, found {len(ranks)}")

    pieces: list[Piece] = []

    # FEN lists ranks from 8 down to 1; rank 1 is rank index 0.
    for i, rank_str in enumerate(ranks):
        rank = 7 - i
        file = 0

        for ch in rank_str:
            if ch.isdigit():
                file += int(ch)
            else:
                idx = PIECE_CHARS.find(ch)
                if idx == -1:
                    raise ValueError(f"invalid piece character '{ch}'")
                if file >= 8:
                    raise ValueError(f"rank '{rank_str}' overflows 8 files")

                pieces.append(Piece(
                    colour = idx // 6,
                    kind   = idx %  6,
                    square = rank * 8 + file,
                ))
                file += 1

        if file != 8:
            raise ValueError(f"rank '{rank_str}' does not sum to 8 files")

    return Position(pieces, stm)


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

def build_accumulators(net: Network, pos: Position) -> tuple[Accumulator, Accumulator]:
    """
    Builds the two perspective accumulators (us = side to move, them = other
    side) for a position, using Chess768's feature layout:

        feature = colour * 384 + piece_kind * 64 + square

    mirroring the square (^ 56) and flipping colour to get the
    opposite-perspective feature.
    """
    us   = net.make_accumulator()
    them = net.make_accumulator()

    for p in pos.pieces:
        white_feat = p.colour * 384 + p.kind * 64 + p.square
        black_feat = (1 - p.colour) * 384 + p.kind * 64 + (p.square ^ 56)

        if pos.stm == 0:          # white to move
            us_feat, them_feat = white_feat, black_feat
        else:                     # black to move
            us_feat, them_feat = black_feat, white_feat

        us.add_feature(us_feat, net)
        them.add_feature(them_feat, net)

    return us, them


# ---------------------------------------------------------------------------
# Top-level FEN evaluation API
# ---------------------------------------------------------------------------

def evaluate_fen(fen: str) -> int:
    """
    Parses `fen`, builds accumulators, and runs the forward pass.
    Returns the evaluation in centipawns, relative to the side to move.
    Mirrors evaluate_fen / evaluate_fen_api in Rust.
    """
    net = Network.load("../checkpoints/quantised_2.bin")
    pos = parse_fen(fen)
    us, them = build_accumulators(net, pos)
    return int(net.evaluate(us, them))

# this one is so we don't have to keep reloading the network parameters
def evaluate_fen(net: Network, fen: str) -> int:
    pos = parse_fen(fen)
    us, them = build_accumulators(net, pos)
    return int(net.evaluate(us, them))

# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # net = Network.load("checkpoints/quantised_2.bin")

    # Starting position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    score = evaluate_fen(fen)
    print(f"Starting position: {score} cp (relative to side to move)")

    # After 1.e4
    fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    score2 = evaluate_fen(fen2)
    print(f"After 1.e4:        {score2} cp (relative to side to move)")
