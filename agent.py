import sys
import random
import numpy as np

# ---------------------------------------------------------------------------
# Step 1: Initialise NNUE weights FIRST (search imports evaluate)
# ---------------------------------------------------------------------------
from src.evaluate import initialize_nnue, PST_TABLE
initialize_nnue()

# ---------------------------------------------------------------------------
# Step 2: Import board and search AFTER weights are loaded
# ---------------------------------------------------------------------------
from src.board import (
    board_from_fen, get_side, get_ep_square, get_castling,
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
)
from src.search import (
    get_best_move,
    compute_hash,
    ZOBRIST_PIECES, ZOBRIST_SIDE, ZOBRIST_CASTLE, ZOBRIST_EP,
    warmup,
)

# ---------------------------------------------------------------------------
# Step 3: JIT warmup (compiles all Numba kernels inside the init budget)
# ---------------------------------------------------------------------------
warmup()

# ---------------------------------------------------------------------------
# OPENING BOOK
# ---------------------------------------------------------------------------
# Keys   : "<piece_placement> <side_to_move>"  (first two FEN fields)
# Values : list of UCI move strings — one is chosen at random each game
# ---------------------------------------------------------------------------
OPENING_BOOK: dict[str, list[str]] = {
    # ── Move 1 ──────────────────────────────────────────────────────────────
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w": ["e2e4", "d2d4"],

    # ── After 1.e4 ──────────────────────────────────────────────────────────
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b": ["e7e5", "c7c5", "e7e6", "c7c6"],
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w": ["g1f3"],
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w": ["g1f3"],
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w": ["d2d4"],
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w": ["d2d4"],
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w": ["e4d5"],
    "rnbqkbnr/ppp1pppp/3p4/8/4P3/8/PPPP1PPP/RNBQKBNR w": ["d2d4"],
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w": ["e4e5"],
    "rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w": ["d2d4"],
    "rnbqkbnr/1ppppppp/p7/8/4P3/8/PPPP1PPP/RNBQKBNR w": ["d2d4"],
    "rnbqkbnr/p1pppppp/1p6/8/4P3/8/PPPP1PPP/RNBQKBNR w": ["d2d4"],
    "rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w": ["e4f5"],
    "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w": ["g1f3"],
    "rnbqkbnr/p1pppppp/8/1p6/4P3/8/PPPP1PPP/RNBQKBNR w": ["d2d4"],
    "rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR b": ["e7e6","c7c6","g8f6"],
    "rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w": ["d2d4"],
    "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w": ["d2d4"],

    # ── After 1.e4 e5 2.Nf3 ─────────────────────────────────────────────────
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w": ["f1b5", "f1c4", "d2d4"],
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w": ["d2d4"],
    "rnbqkbnr/pppp2pp/8/4pp2/4P3/5N2/PPPP1PPP/RNBQKB1R w": ["f3e5"],

    # ── After 1.e4 e5 2.Nf3 Nc6 3.Bb5 (Ruy Lopez) ──────────────────────────
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b": ["a7a6", "g8f6"],
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w": ["b5a4"],
    "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w": ["e1g1"],
    "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQR3 b": ["f8e7"],

    # ── After 1.e4 e5 2.Nf3 Nc6 3.Bc4 (Italian) ────────────────────────────
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b": ["f8c5", "g8f6"],
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w": ["c2c3"],
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R b": ["g8f6"],
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w": ["d2d3"],

    # ── After 1.e4 e5 2.Nf3 Nc6 3.d4 (Scotch) ──────────────────────────────
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b": ["e5d4"],
    "r1bqkbnr/pppp1ppp/2n5/8/3pP3/5N2/PPP2PPP/RNBQKB1R w": ["f3d4"],

    # ── After 1.e4 c5 2.Nf3 (Sicilian) ─────────────────────────────────────
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b": ["d7d6","e7e6","b8c6"],
    "rnbqkbnr/pp1p1ppp/4p3/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w": ["d2d4"],
    "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w": ["d2d4"],
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w": ["b1c3"],
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w": ["d2d4"],

    # ── After 1.e4 e6 2.d4 (French) ─────────────────────────────────────────
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w": ["d2d4"],
    "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b": ["d7d5"],
    "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w": ["b1c3"],
    "rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w": ["c1g5"],

    # ── After 1.e4 c6 2.d4 (Caro-Kann) ─────────────────────────────────────
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w": ["d2d4"],
    "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b": ["d7d5"],
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w": ["b1c3"],

    # ── After 1.d4 ──────────────────────────────────────────────────────────
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b": ["d7d5","g8f6","e7e6","f7f5"],
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w": ["c2c4"],
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w": ["c2c4"],
    "rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w": ["c2c4"],
    "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR w": ["c2c4"],
    "rnbqkbnr/pppppp1p/6p1/8/3P4/8/PPP1PPPP/RNBQKBNR w": ["c2c4"],
    "rnbqkbnr/ppp1pppp/3p4/8/3P4/8/PPP1PPPP/RNBQKBNR w": ["c2c4"],
    "rnbqkbnr/p1pppppp/1p6/8/3P4/8/PPP1PPPP/RNBQKBNR w": ["c2c4"],
    "rnbqkbnr/pp1ppppp/2p5/8/3P4/8/PPP1PPPP/RNBQKBNR w": ["c2c4"],
    "r1bqkbnr/pppppppp/2n5/8/3P4/8/PPP1PPPP/RNBQKBNR w": ["d4d5"],

    # ── After 1.d4 d5 2.c4 (Queen's Gambit) ────────────────────────────────
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b": ["e7e6", "c7c6", "g8f6"],
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w": ["b1c3"],
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w": ["c1g5"],
    "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w": ["g1f3"],
    "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w": ["b1c3"],

    # ── After 1.d4 Nf6 2.c4 (King's Indian / Nimzo / Grunfeld) ─────────────
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b": ["g7g6", "e7e6", "d7d5"],
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w": ["b1c3"],
    "rnbqk2r/ppppppbp/5np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR b": ["d7d6"],
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w": ["b1c3"],
    "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w": ["b1c3"],

    # ── After 1.c4 (English) ────────────────────────────────────────────────
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b": ["e7e5", "c7c5", "g8f6"],
    "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w": ["b1c3"],
    "rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w": ["b1c3"],

    # ── After 1.Nf3 ─────────────────────────────────────────────────────────
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b": ["d7d5","g8f6","c7c5"],
    "rnbqkbnr/ppp1pppp/8/3p4/8/5N2/PPPPPPPP/RNBQKB1R w": ["d2d4"],
}


def _book_lookup(fen: str) -> str | None:
    """Return a random book move for the position, or None if not in book."""
    key = " ".join(fen.split()[:2])
    moves = OPENING_BOOK.get(key)
    if moves:
        return random.choice(moves)
    return None


# ---------------------------------------------------------------------------
# GAME HISTORY  (Zobrist hashes of positions already played this game)
# ---------------------------------------------------------------------------
_game_history: list = []
_last_fullmove: int = 0
_last_side: int     = -1


def _is_new_game(halfmove: int, fullmove: int, side: int) -> bool:
    global _last_fullmove, _last_side
    if _last_fullmove == 0:
        return True
    if fullmove < _last_fullmove:
        return True
    if fullmove == 1 and halfmove == 0 and side == 0:
        return True
    return False


def _move_int_to_uci(move_int: int) -> str:
    FILES = "abcdefgh"
    from_sq = move_int & 63
    to_sq   = (move_int >> 6) & 63
    promo   = (move_int >> 12) & 15

    from_uci = FILES[from_sq % 8] + str(from_sq // 8 + 1)
    to_uci   = FILES[to_sq   % 8] + str(to_sq   // 8 + 1)

    PROMO_MAP = {
        W_QUEEN: "q", W_ROOK: "r", W_BISHOP: "b", W_KNIGHT: "n",
        B_QUEEN: "q", B_ROOK: "r", B_BISHOP: "b", B_KNIGHT: "n",
    }
    promo_str = PROMO_MAP.get(promo, "")
    return from_uci + to_uci + promo_str


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def get_move(fen: str, time_left_ms: int) -> str:
    global _game_history, _last_fullmove, _last_side

    # ── Parse FEN fields ─────────────────────────────────────────────────────
    parts     = fen.split()
    side_char = parts[1] if len(parts) > 1 else "w"
    halfmove  = int(parts[4]) if len(parts) > 4 else 0
    fullmove  = int(parts[5]) if len(parts) > 5 else 1
    side      = 0 if side_char == "w" else 1

    # ── New game detection ────────────────────────────────────────────────────
    if _is_new_game(halfmove, fullmove, side):
        print("New game detected — history reset.")
        _game_history = []

    _last_fullmove = fullmove
    _last_side     = side

    # ── Trim history to last 100 entries (50-move rule) ───────────────────────
    if len(_game_history) > 100:
        _game_history = _game_history[-100:]

    print(f"Searching position: {fen[:60]}...")
    print(f"Time remaining: {time_left_ms}ms")
    print(f"History depth: {len(_game_history)} positions")

    # ── Build board ───────────────────────────────────────────────────────────
    board = board_from_fen(fen)

    # ── Compute current position hash ─────────────────────────────────────────
    current_hash = np.uint64(
        compute_hash(board, ZOBRIST_PIECES, ZOBRIST_SIDE, ZOBRIST_CASTLE, ZOBRIST_EP)
    )

    # ── Opening book lookup ───────────────────────────────────────────────────
    book_move = _book_lookup(fen)
    if book_move:
        print(f"Book move: {book_move}")
        # Still record hash for repetition tracking
        _game_history.append(current_hash)
        return book_move

    # ── Search ────────────────────────────────────────────────────────────────
    move_int = get_best_move(board, time_left_ms, game_hist=_game_history)

    # ── Record hash AFTER search (never before — prevents a1a1 bug) ───────────
    _game_history.append(current_hash)

    uci = _move_int_to_uci(int(move_int))
    print(f"Best move: {uci}", file=sys.stderr)
    return uci
