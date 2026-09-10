import time
import numpy as np
from numba import njit, uint64, int32
from numba.typed import Dict
from numba.core import types

from src.board import (
    copy_board, get_side, piece_on, encode_move,
    get_pieces, get_white_occ, get_black_occ, get_all_occ,
    get_ep_square, get_castling, set_bit, clear_bit, get_bit,
    pop_lsb, lsb,
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ,
)
from src.evaluate import evaluate

# =============================================================================
# ATTACK HELPERS
# =============================================================================

@njit(cache=True)
def knight_attacks_sq(sq):
    bb = uint64(0)
    r, f = sq // 8, sq % 8
    for dr, df in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        nr, nf = r + dr, f + df
        if 0 <= nr <= 7 and 0 <= nf <= 7:
            bb = set_bit(bb, nr * 8 + nf)
    return bb

@njit(cache=True)
def king_attacks_sq(sq):
    bb = uint64(0)
    r, f = sq // 8, sq % 8
    for dr in range(-1, 2):
        for df in range(-1, 2):
            if dr == 0 and df == 0:
                continue
            nr, nf = r + dr, f + df
            if 0 <= nr <= 7 and 0 <= nf <= 7:
                bb = set_bit(bb, nr * 8 + nf)
    return bb

@njit(cache=True)
def bishop_attacks_sq(sq, occ):
    bb = uint64(0)
    r, f = sq // 8, sq % 8
    for dr, df in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nf = r + dr, f + df
        while 0 <= nr <= 7 and 0 <= nf <= 7:
            s = nr * 8 + nf
            bb = set_bit(bb, s)
            if get_bit(occ, s):
                break
            nr += dr; nf += df
    return bb

@njit(cache=True)
def rook_attacks_sq(sq, occ):
    bb = uint64(0)
    r, f = sq // 8, sq % 8
    for dr, df in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nf = r + dr, f + df
        while 0 <= nr <= 7 and 0 <= nf <= 7:
            s = nr * 8 + nf
            bb = set_bit(bb, s)
            if get_bit(occ, s):
                break
            nr += dr; nf += df
    return bb

@njit(cache=True)
def queen_attacks_sq(sq, occ):
    return bishop_attacks_sq(sq, occ) | rook_attacks_sq(sq, occ)

@njit(cache=True)
def is_square_attacked(board, sq, by_side):
    """True if `sq` is attacked by `by_side` (0=white, 1=black)."""
    occ = get_all_occ(board)
    if by_side == 0:
        if knight_attacks_sq(sq) & get_pieces(board, W_KNIGHT): return True
        if king_attacks_sq(sq)   & get_pieces(board, W_KING):   return True
        if bishop_attacks_sq(sq, occ) & (get_pieces(board, W_BISHOP) | get_pieces(board, W_QUEEN)): return True
        if rook_attacks_sq(sq, occ)   & (get_pieces(board, W_ROOK)   | get_pieces(board, W_QUEEN)): return True
        r, f = sq // 8, sq % 8
        if r > 0:
            if f > 0 and get_bit(get_pieces(board, W_PAWN), (r-1)*8+(f-1)): return True
            if f < 7 and get_bit(get_pieces(board, W_PAWN), (r-1)*8+(f+1)): return True
    else:
        if knight_attacks_sq(sq) & get_pieces(board, B_KNIGHT): return True
        if king_attacks_sq(sq)   & get_pieces(board, B_KING):   return True
        if bishop_attacks_sq(sq, occ) & (get_pieces(board, B_BISHOP) | get_pieces(board, B_QUEEN)): return True
        if rook_attacks_sq(sq, occ)   & (get_pieces(board, B_ROOK)   | get_pieces(board, B_QUEEN)): return True
        r, f = sq // 8, sq % 8
        if r < 7:
            if f > 0 and get_bit(get_pieces(board, B_PAWN), (r+1)*8+(f-1)): return True
            if f < 7 and get_bit(get_pieces(board, B_PAWN), (r+1)*8+(f+1)): return True
    return False

# =============================================================================
# MAKE MOVE
# =============================================================================

@njit(cache=True)
def make_move(board, move):
    from_sq = move & 63
    to_sq   = (move >> 6) & 63
    promo   = (move >> 12) & 15

    piece = piece_on(board, from_sq)
    if piece == -1:
        return

    side  = get_side(board)
    ep_sq = get_ep_square(board)

    # Lift piece off source
    board[piece] = clear_bit(board[piece], from_sq)
    occ_src = 12 if piece < 6 else 13
    board[occ_src] = clear_bit(board[occ_src], from_sq)

    # Remove any captured piece
    captured = piece_on(board, to_sq)
    if captured != -1:
        board[captured] = clear_bit(board[captured], to_sq)
        occ_cap = 12 if captured < 6 else 13
        board[occ_cap] = clear_bit(board[occ_cap], to_sq)

    # En passant capture
    new_ep = 64
    if (piece == W_PAWN or piece == B_PAWN) and to_sq == ep_sq and ep_sq != 64:
        if side == 0:
            ep_cap = to_sq - 8
            board[B_PAWN] = clear_bit(board[B_PAWN], ep_cap)
            board[13]     = clear_bit(board[13], ep_cap)
        else:
            ep_cap = to_sq + 8
            board[W_PAWN] = clear_bit(board[W_PAWN], ep_cap)
            board[12]     = clear_bit(board[12], ep_cap)

    # Place piece on destination (promotion replaces piece type)
    landing = piece if promo == 0 else promo
    board[landing] = set_bit(board[landing], to_sq)
    occ_dst = 12 if landing < 6 else 13
    board[occ_dst] = set_bit(board[occ_dst], to_sq)

    # Set new EP square for double pawn pushes
    if piece == W_PAWN and to_sq - from_sq == 16:
        new_ep = from_sq + 8
    elif piece == B_PAWN and from_sq - to_sq == 16:
        new_ep = from_sq - 8

    # Castling: move the rook
    castling = get_castling(board)
    if piece == W_KING:
        if from_sq == 4 and to_sq == 6:
            board[W_ROOK] = clear_bit(board[W_ROOK], 7)
            board[12]     = clear_bit(board[12], 7)
            board[W_ROOK] = set_bit(board[W_ROOK], 5)
            board[12]     = set_bit(board[12], 5)
        elif from_sq == 4 and to_sq == 2:
            board[W_ROOK] = clear_bit(board[W_ROOK], 0)
            board[12]     = clear_bit(board[12], 0)
            board[W_ROOK] = set_bit(board[W_ROOK], 3)
            board[12]     = set_bit(board[12], 3)
        castling &= ~(int(CASTLE_WK) | int(CASTLE_WQ))
    elif piece == B_KING:
        if from_sq == 60 and to_sq == 62:
            board[B_ROOK] = clear_bit(board[B_ROOK], 63)
            board[13]     = clear_bit(board[13], 63)
            board[B_ROOK] = set_bit(board[B_ROOK], 61)
            board[13]     = set_bit(board[13], 61)
        elif from_sq == 60 and to_sq == 58:
            board[B_ROOK] = clear_bit(board[B_ROOK], 56)
            board[13]     = clear_bit(board[13], 56)
            board[B_ROOK] = set_bit(board[B_ROOK], 59)
            board[13]     = set_bit(board[13], 59)
        castling &= ~(int(CASTLE_BK) | int(CASTLE_BQ))

    # Revoke castling rights when rooks move or are captured
    if from_sq == 0  or to_sq == 0:  castling &= ~int(CASTLE_WQ)
    if from_sq == 7  or to_sq == 7:  castling &= ~int(CASTLE_WK)
    if from_sq == 56 or to_sq == 56: castling &= ~int(CASTLE_BQ)
    if from_sq == 63 or to_sq == 63: castling &= ~int(CASTLE_BK)

    board[14] = board[12] | board[13]
    board[15] = uint64(new_ep)
    board[16] = uint64(castling)
    board[17] = uint64(1 - side)
    board[18] = uint64(0) if (captured != -1 or piece == W_PAWN or piece == B_PAWN) \
                          else board[18] + uint64(1)
    if side == 1:
        board[19] = board[19] + uint64(1)

@njit(cache=True)
def make_null_move(board):
    board[17] = uint64(1 - int(board[17]))
    board[15] = uint64(64)

# =============================================================================
# LEGAL MOVE GENERATION
# =============================================================================

@njit(cache=True)
def _add_moves(moves, count, from_sq, targets, promo=0):
    bb = targets
    while bb:
        sq, bb = pop_lsb(bb)
        moves[count] = (promo << 12) | (sq << 6) | from_sq
        count += 1
    return count

@njit(cache=True)
def generate_legal_moves(board):
    side = get_side(board)
    occ  = get_all_occ(board)
    my_occ  = get_white_occ(board) if side == 0 else get_black_occ(board)
    opp_occ = get_black_occ(board) if side == 0 else get_white_occ(board)
    ep_sq   = get_ep_square(board)
    castling = get_castling(board)

    pseudo = np.zeros(256, dtype=np.int32)
    count  = 0

    if side == 0:  # ---- WHITE ----
        # Pawns
        pawns = get_pieces(board, W_PAWN)
        while pawns:
            sq, pawns = pop_lsb(pawns)
            r, f = sq // 8, sq % 8
            # Single push
            if r < 7 and not get_bit(occ, sq + 8):
                if r == 6:  # promotion
                    for p in (W_QUEEN, W_ROOK, W_BISHOP, W_KNIGHT):
                        pseudo[count] = (p << 12) | ((sq+8) << 6) | sq; count += 1
                else:
                    pseudo[count] = ((sq+8) << 6) | sq; count += 1
                    # Double push
                    if r == 1 and not get_bit(occ, sq + 16):
                        pseudo[count] = ((sq+16) << 6) | sq; count += 1
            # Captures
            for df in (-1, 1):
                nf = f + df
                if 0 <= nf <= 7:
                    tsq = (r+1)*8 + nf
                    if get_bit(opp_occ, tsq):
                        if r == 6:
                            for p in (W_QUEEN, W_ROOK, W_BISHOP, W_KNIGHT):
                                pseudo[count] = (p << 12) | (tsq << 6) | sq; count += 1
                        else:
                            pseudo[count] = (tsq << 6) | sq; count += 1
                    elif tsq == ep_sq and ep_sq != 64:
                        pseudo[count] = (tsq << 6) | sq; count += 1

        # Knights
        knights = get_pieces(board, W_KNIGHT)
        while knights:
            sq, knights = pop_lsb(knights)
            targets = knight_attacks_sq(sq) & ~my_occ
            count = _add_moves(pseudo, count, sq, targets)

        # Bishops
        bishops = get_pieces(board, W_BISHOP)
        while bishops:
            sq, bishops = pop_lsb(bishops)
            targets = bishop_attacks_sq(sq, occ) & ~my_occ
            count = _add_moves(pseudo, count, sq, targets)

        # Rooks
        rooks = get_pieces(board, W_ROOK)
        while rooks:
            sq, rooks = pop_lsb(rooks)
            targets = rook_attacks_sq(sq, occ) & ~my_occ
            count = _add_moves(pseudo, count, sq, targets)

        # Queens
        queens = get_pieces(board, W_QUEEN)
        while queens:
            sq, queens = pop_lsb(queens)
            targets = queen_attacks_sq(sq, occ) & ~my_occ
            count = _add_moves(pseudo, count, sq, targets)

        # King
        ksq = lsb(get_pieces(board, W_KING))
        if ksq < 64:
            targets = king_attacks_sq(ksq) & ~my_occ
            count = _add_moves(pseudo, count, ksq, targets)
            # Castling
            if castling & int(CASTLE_WK):
                if not get_bit(occ, 5) and not get_bit(occ, 6):
                    if not is_square_attacked(board, 4, 1) and \
                       not is_square_attacked(board, 5, 1) and \
                       not is_square_attacked(board, 6, 1):
                        pseudo[count] = (6 << 6) | 4; count += 1
            if castling & int(CASTLE_WQ):
                if not get_bit(occ, 3) and not get_bit(occ, 2) and not get_bit(occ, 1):
                    if not is_square_attacked(board, 4, 1) and \
                       not is_square_attacked(board, 3, 1) and \
                       not is_square_attacked(board, 2, 1):
                        pseudo[count] = (2 << 6) | 4; count += 1

    else:  # ---- BLACK ----
        # Pawns
        pawns = get_pieces(board, B_PAWN)
        while pawns:
            sq, pawns = pop_lsb(pawns)
            r, f = sq // 8, sq % 8
            # Single push
            if r > 0 and not get_bit(occ, sq - 8):
                if r == 1:  # promotion
                    for p in (B_QUEEN, B_ROOK, B_BISHOP, B_KNIGHT):
                        pseudo[count] = (p << 12) | ((sq-8) << 6) | sq; count += 1
                else:
                    pseudo[count] = ((sq-8) << 6) | sq; count += 1
                    # Double push
                    if r == 6 and not get_bit(occ, sq - 16):
                        pseudo[count] = ((sq-16) << 6) | sq; count += 1
            # Captures
            for df in (-1, 1):
                nf = f + df
                if 0 <= nf <= 7:
                    tsq = (r-1)*8 + nf
                    if tsq >= 0 and get_bit(opp_occ, tsq):
                        if r == 1:
                            for p in (B_QUEEN, B_ROOK, B_BISHOP, B_KNIGHT):
                                pseudo[count] = (p << 12) | (tsq << 6) | sq; count += 1
                        else:
                            pseudo[count] = (tsq << 6) | sq; count += 1
                    elif tsq == ep_sq and ep_sq != 64:
                        pseudo[count] = (tsq << 6) | sq; count += 1

        # Knights
        knights = get_pieces(board, B_KNIGHT)
        while knights:
            sq, knights = pop_lsb(knights)
            targets = knight_attacks_sq(sq) & ~my_occ
            count = _add_moves(pseudo, count, sq, targets)

        # Bishops
        bishops = get_pieces(board, B_BISHOP)
        while bishops:
            sq, bishops = pop_lsb(bishops)
            targets = bishop_attacks_sq(sq, occ) & ~my_occ
            count = _add_moves(pseudo, count, sq, targets)

        # Rooks
        rooks = get_pieces(board, B_ROOK)
        while rooks:
            sq, rooks = pop_lsb(rooks)
            targets = rook_attacks_sq(sq, occ) & ~my_occ
            count = _add_moves(pseudo, count, sq, targets)

        # Queens
        queens = get_pieces(board, B_QUEEN)
        while queens:
            sq, queens = pop_lsb(queens)
            targets = queen_attacks_sq(sq, occ) & ~my_occ
            count = _add_moves(pseudo, count, sq, targets)

        # King
        ksq = lsb(get_pieces(board, B_KING))
        if ksq < 64:
            targets = king_attacks_sq(ksq) & ~my_occ
            count = _add_moves(pseudo, count, ksq, targets)
            # Castling
            if castling & int(CASTLE_BK):
                if not get_bit(occ, 61) and not get_bit(occ, 62):
                    if not is_square_attacked(board, 60, 0) and \
                       not is_square_attacked(board, 61, 0) and \
                       not is_square_attacked(board, 62, 0):
                        pseudo[count] = (62 << 6) | 60; count += 1
            if castling & int(CASTLE_BQ):
                if not get_bit(occ, 59) and not get_bit(occ, 58) and not get_bit(occ, 57):
                    if not is_square_attacked(board, 60, 0) and \
                       not is_square_attacked(board, 59, 0) and \
                       not is_square_attacked(board, 58, 0):
                        pseudo[count] = (58 << 6) | 60; count += 1

    # --- Legality filter: remove moves that leave king in check ---
    king_piece = W_KING if side == 0 else B_KING
    legal = np.zeros(256, dtype=np.int32)
    legal_count = 0
    for i in range(count):
        mv = pseudo[i]
        nb = copy_board(board)
        make_move(nb, mv)
        ksq = lsb(get_pieces(nb, king_piece))
        if ksq < 64 and not is_square_attacked(nb, ksq, 1 - side):
            legal[legal_count] = mv
            legal_count += 1

    return legal[:legal_count]

# =============================================================================
# ZOBRIST HASHING
# =============================================================================
np.random.seed(42)
ZOBRIST_PIECES = np.random.randint(0, 2**63, size=(12, 64), dtype=np.int64).view(np.uint64)
ZOBRIST_SIDE   = np.random.randint(0, 2**63, dtype=np.int64).view(np.uint64)[()]

@njit(cache=True)
def compute_hash(board):
    h = uint64(0)
    for sq in range(64):
        p = piece_on(board, sq)
        if p != -1:
            h ^= ZOBRIST_PIECES[p, sq]
    if get_side(board) == 1:
        h ^= ZOBRIST_SIDE
    return h

# =============================================================================
# TRANSPOSITION TABLE
# =============================================================================
INFINITY = 999999
FLAG_EXACT      = 1
FLAG_LOWERBOUND = 2
FLAG_UPPERBOUND = 3

def create_tt():
    return Dict.empty(
        key_type=types.uint64,
        value_type=types.UniTuple(types.int32, 4),
    )

# ---------------------------------------------------------------------------
# THE FIX: create_tt() is called INSIDE get_best_move() every time, so the
# TT is always fresh and never carries stale/corrupted data between games.
# A module-level TT was the root cause: after the first game the table was
# full of entries from a completely different position tree, causing the
# engine to return garbage moves (or no move at all) and crash.
# ---------------------------------------------------------------------------

@njit(cache=True)
def tt_store(tt, h, depth, score, flag, best_move):
    tt[h] = (int32(depth), int32(score), int32(flag), int32(best_move))

@njit(cache=True)
def tt_probe(tt, h, depth, alpha, beta):
    if h not in tt:
        return False, int32(0), int32(0)
    stored_depth, stored_score, stored_flag, stored_move = tt[h]
    if stored_depth < depth:
        return False, int32(0), int32(stored_move)
    if stored_flag == 1:                          # EXACT
        return True, int32(stored_score), int32(stored_move)
    elif stored_flag == 2 and stored_score >= beta:   # LOWER
        return True, int32(stored_score), int32(stored_move)
    elif stored_flag == 3 and stored_score <= alpha:  # UPPER
        return True, int32(stored_score), int32(stored_move)
    return False, int32(0), int32(stored_move)

# =============================================================================
# NEGAMAX + ALPHA-BETA
# =============================================================================

@njit(cache=True)
def negamax(board, depth, alpha, beta, color, tt, allow_null):
    h = compute_hash(board)
    hit, tt_score, tt_move = tt_probe(tt, h, depth, alpha, beta)
    if hit:
        return tt_score

    if depth <= 0:
        # evaluate() returns score from side-to-move perspective
        return evaluate(board)

    # Null-move pruning
    if allow_null and depth >= 3:
        null_board = copy_board(board)
        make_null_move(null_board)
        null_score = -negamax(null_board, depth - 3, -beta, -beta + 1, -color, tt, False)
        if null_score >= beta:
            return beta

    moves = generate_legal_moves(board)
    if len(moves) == 0:
        # No legal moves: checkmate or stalemate
        king_piece = 5 if get_side(board) == 0 else 11
        ksq = lsb(get_pieces(board, king_piece))
        if ksq < 64 and is_square_attacked(board, ksq, 1 - get_side(board)):
            return -INFINITY + depth   # Checkmate (prefer faster mates)
        return 0                       # Stalemate

    best_score  = -INFINITY
    best_move   = int32(0)
    orig_alpha  = alpha

    for i in range(len(moves)):
        move = moves[i]
        nb   = copy_board(board)
        make_move(nb, move)

        if i == 0:
            score = -negamax(nb, depth - 1, -beta, -alpha, -color, tt, True)
        else:
            score = -negamax(nb, depth - 1, -alpha - 1, -alpha, -color, tt, True)
            if alpha < score < beta:
                score = -negamax(nb, depth - 1, -beta, -alpha, -color, tt, True)

        if score > best_score:
            best_score = score
            best_move  = int32(move)

        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    flag = 1 if orig_alpha < best_score < beta else (2 if best_score >= beta else 3)
    tt_store(tt, h, depth, best_score, flag, best_move)
    return best_score

# =============================================================================
# ITERATIVE DEEPENING + TIME MANAGEMENT
# =============================================================================

def get_best_move(board_array, time_left_ms):
    """
    Iterative-deepening search with per-move time budget.

    A brand-new TT is created for every call so stale entries from
    previous games can never corrupt the search.
    """
    # Fresh TT every game — this is the key fix for the cross-game crash
    tt = create_tt()

    start   = time.time()
    # Allocate ~1/40th of remaining time, clamped to [0.5s, 8s]
    budget  = max(0.5, min(8.0, (time_left_ms / 1000.0) / 40.0))

    def elapsed():
        return time.time() - start

    def time_ok(fraction=1.0):
        return elapsed() < budget * fraction

    last_best  = 0
    last_score = 0

    for depth in range(1, 20):
        if not time_ok(0.5) and depth > 1:
            break  # Don't start a depth we can't finish

        score = negamax(board_array, depth, -INFINITY, INFINITY, 1, tt, True)

        # Read best move from TT root entry
        h = compute_hash(board_array)
        current_best = 0
        if h in tt:
            _, _, _, current_best = tt[h]

        last_best  = current_best if current_best != 0 else last_best
        last_score = score

        #print(f"  depth {depth:2d} | score {score:+6d} | "f"move {current_best} | t={elapsed():.2f}s")

        if not time_ok():
            break

        # Stop early on forced mate
        if abs(score) > INFINITY - 100:
            break

    return int(last_best)
