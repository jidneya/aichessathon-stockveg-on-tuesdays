import os
import shutil
import chess
import chess.polyglot
import chess.syzygy
from src.board import board_from_fen, decode_move, copy_board
from src.search import get_best_move, negamax, create_tt, compute_hash, make_move, clear_tt

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
# PATHS — adjust these to wherever you place your book/tablebase files
# =============================================================================
_BASE        = os.path.dirname(os.path.abspath(__file__))
_BOOK_PATH   = os.path.join(_BASE, "src", "books",  "opening.bin")   # polyglot book
_SYZYGY_PATH = os.path.join(_BASE, "src", "syzygy")                  # folder of .rtbw/.rtbz files
_NNUE_PATH   = os.path.join(_BASE, "src", "checkpoints", "final_final_weights.bin")

# =============================================================================
# NNUE INITIALISATION  (runs once per process)
# =============================================================================
print("Initializing NNUE evaluation...", flush=True)
initialize_nnue(_NNUE_PATH)
print("NNUE ready!", flush=True)

# =============================================================================
# OPENING BOOK — load once, keep open for the lifetime of the process.
# chess.polyglot.open_reader() is a lightweight memory-mapped reader;
# keeping it open avoids re-opening the file on every move.
# =============================================================================
_book_reader = None
if os.path.isfile(_BOOK_PATH):
    _book_reader = chess.polyglot.open_reader(_BOOK_PATH)
    print(f"Opening book loaded: {_BOOK_PATH}", flush=True)
else:
    print(f"WARNING: No opening book found at {_BOOK_PATH} — skipping.", flush=True)

# =============================================================================
# SYZYGY TABLEBASES — open the tablebase directory once at startup.
# chess.syzygy.open_tablebase() scans the folder for .rtbw / .rtbz files.
# =============================================================================
_tablebase = None
if os.path.isdir(_SYZYGY_PATH):
    _tablebase = chess.syzygy.open_tablebase(_SYZYGY_PATH)
    print(f"Syzygy tablebases loaded from: {_SYZYGY_PATH}", flush=True)
else:
    print(f"WARNING: No Syzygy tablebase folder at {_SYZYGY_PATH} — skipping.", flush=True)

# =============================================================================
# NUMBA WARM-UP — forces JIT compilation of all search kernels NOW,
# during the 90-second pre-game window, so move 1 is instant.
# =============================================================================
print("Warming up Numba JIT (compiling search kernels)...", flush=True)
_STARTPOS    = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_warmup_board = board_from_fen(_STARTPOS)
get_best_move(_warmup_board, time_left_ms=5000, game_hist=[])
print("Warm-up complete — ready to play!", flush=True)


# =============================================================================
# GAME HISTORY
# =============================================================================
_game_history = []
_last_fullmove = 0

# =============================================================================
# HELPERS
# =============================================================================

def _try_book_move(board: chess.Board) -> str | None:
    """
    Look up the current position in the polyglot opening book.
    Returns the best-weighted UCI move string, or None if not found.

    chess.polyglot.open_reader.weighted_choice(board) picks a move
    proportionally to its book weight (higher weight = stronger/more common).
    Use .find(board) instead if you always want the single highest-weight entry.
    """
    if _book_reader is None:
        return None
    try:
        entry = _book_reader.weighted_choice(board)   # or .find(board) for deterministic
        uci = entry.move.uci()
        print(f"[Book] {uci}", flush=True)
        return uci
    except IndexError:
        # Position not in book
        return None


def _try_tablebase_move(board: chess.Board) -> str | None:
    """
    Query the Syzygy tablebases for a perfect endgame move.

    Strategy:
      1. probe_wdl()  — is this position a Win / Draw / Loss?
      2. If it's a win, iterate legal moves, probe each resulting position,
         and pick any move that keeps us in a winning WDL.
      3. If it's a draw, pick a move that keeps us in a draw (avoids loss).

    We only enter this path when ≤ 6 pieces are on the board (Syzygy limit).
    """
    if _tablebase is None:
        return None

    # Only probe when piece count is within tablebase range
    piece_count = chess.popcount(board.occupied)
    if piece_count > 6:
        return None

    try:
        wdl = _tablebase.probe_wdl(board)
    except chess.syzygy.MissingTableError:
        return None

    # wdl > 0  → current side is winning
    # wdl == 0 → drawn
    # wdl < 0  → losing (don't need to handle specially — just fall through)

    best_move = None
    best_wdl  = -3   # worst possible

    for move in board.legal_moves:
        board.push(move)
        try:
            # Opponent's WDL after our move — negate for our perspective
            opp_wdl = _tablebase.probe_wdl(board)
            our_wdl = -opp_wdl
        except chess.syzygy.MissingTableError:
            board.pop()
            continue

        board.pop()

        if our_wdl > best_wdl:
            best_wdl  = our_wdl
            best_move = move

    if best_move is not None:
        uci = best_move.uci()
        wdl_label = {2: "Win", 1: "Cursed Win", 0: "Draw",
                     -1: "Blessed Loss", -2: "Loss"}.get(best_wdl, "?")
        print(f"[Tablebase] {uci}  WDL={wdl_label}", flush=True)
        return uci

    return None


def _encoded_to_uci(best_encoded_move: int) -> str:
    """Convert your internal packed move integer to a UCI string."""
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

    return uci


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def get_move(fen: str, time_left_ms: int) -> str:
    global _game_history, _last_fullmove
    """
    Called by the harness for every move.
    Returns a UCI move string e.g. 'e2e4', 'e7e8q'.

    Priority order:
      1. Opening book  (instant, no clock cost)
      2. Syzygy tablebases  (perfect endgame play, ≤6 pieces)
      3. NNUE + negamax search  (main engine)
    """
    print(f"Searching position: {fen[:50]}...", flush=True)
    print(f"Time remaining: {time_left_ms}ms", flush=True)

    # Parse FEN fields to track game progress
    parts = fen.split()
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1

    # Detect new game to reset history and memory
    if fullmove < _last_fullmove or (fullmove == 1 and halfmove == 0):
        print("New game detected — history and TT reset.", flush=True)
        _game_history = []
        clear_tt()

    # Build a python-chess Board from the FEN — used by polyglot & syzygy only.
    # Your Numba search uses board_from_fen() separately below.
    board = chess.Board(fen)
    board_array = board_from_fen(fen)
    current_hash = np.uint64(compute_hash(board_array))

    # ------------------------------------------------------------------
    # 1. Opening book
    # ------------------------------------------------------------------
    book_move = _try_book_move(board)
    if book_move:
        _game_history.append(current_hash)
        return book_move

    # ------------------------------------------------------------------
    # 2. Syzygy tablebases (endgame, ≤6 pieces)
    # ------------------------------------------------------------------
    tb_move = _try_tablebase_move(board)
    if tb_move:
        _game_history.append(current_hash)
        return tb_move

    # ------------------------------------------------------------------
    # 3. NNUE + negamax search (main engine)
    # ------------------------------------------------------------------
    _game_history.append(current_hash)
    best_encoded_move = get_best_move(board_array, time_left_ms, _game_history)

    # Apply the engine's move to record the opponent's resulting position 
    child = copy_board(board_array)
    make_move(child, best_encoded_move)
    _game_history.append(np.uint64(compute_hash(child)))

    # Trim history to prevent memory leak in excessively long games
    if len(_game_history) > 200:
        _game_history = _game_history[-200:]
    
    uci = _encoded_to_uci(best_encoded_move)
    print(f"Best move: {uci}", flush=True)
    return uci
