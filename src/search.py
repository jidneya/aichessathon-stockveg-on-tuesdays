import time
import shutil
import os
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
from src.evaluate import nnue_forward, L0_WEIGHTS, L0_BIASES, L1_WEIGHTS, L1_BIAS

INFINITY = 999999

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

    # Remove piece from source
    board[piece] = clear_bit(board[piece], from_sq)
    if piece < 6:
        board[12] = clear_bit(board[12], from_sq)
    else:
        board[13] = clear_bit(board[13], from_sq)

    # Remove any captured piece on destination
    captured = piece_on(board, to_sq)
    if captured != -1:
        board[captured] = clear_bit(board[captured], to_sq)
        if captured < 6:
            board[12] = clear_bit(board[12], to_sq)
        else:
            board[13] = clear_bit(board[13], to_sq)

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

    # Determine landing piece (promotion replaces pawn)
    if promo != 0:
        landing = promo
    else:
        landing = piece

    # Place landing piece on destination
    board[landing] = set_bit(board[landing], to_sq)
    if landing < 6:
        board[12] = set_bit(board[12], to_sq)
    else:
        board[13] = set_bit(board[13], to_sq)

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

    if from_sq == 0  or to_sq == 0:  castling &= ~int(CASTLE_WQ)
    if from_sq == 7  or to_sq == 7:  castling &= ~int(CASTLE_WK)
    if from_sq == 56 or to_sq == 56: castling &= ~int(CASTLE_BQ)
    if from_sq == 63 or to_sq == 63: castling &= ~int(CASTLE_BK)

    board[14] = board[12] | board[13]
    board[15] = uint64(new_ep)
    board[16] = uint64(castling)
    board[17] = uint64(1 - side)
    if captured != -1 or piece == W_PAWN or piece == B_PAWN:
        board[18] = uint64(0)
    else:
        board[18] = board[18] + uint64(1)
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
def generate_legal_moves(board):
    side = get_side(board)
    occ  = get_all_occ(board)
    my_occ  = get_white_occ(board) if side == 0 else get_black_occ(board)
    opp_occ = get_black_occ(board) if side == 0 else get_white_occ(board)
    ep_sq   = get_ep_square(board)
    castling = get_castling(board)

    buf = np.zeros(256, dtype=np.int32)
    n   = 0

    if side == 0:
        # ── White pawns ──────────────────────────────────────────────────────
        pawns = get_pieces(board, W_PAWN)
        while pawns:
            sq, pawns = pop_lsb(pawns)
            r = sq // 8
            # single push
            if not get_bit(occ, sq + 8):
                if r == 6:   # promotion
                    for p in (W_QUEEN, W_ROOK, W_BISHOP, W_KNIGHT):
                        buf[n] = encode_move(sq, sq+8, p); n += 1
                else:
                    buf[n] = encode_move(sq, sq+8); n += 1
                    if r == 1 and not get_bit(occ, sq+16):
                        buf[n] = encode_move(sq, sq+16); n += 1
            # captures
            for to_sq in (sq+7, sq+9):
                if to_sq > 63: continue
                tf = to_sq % 8
                if abs(tf - sq%8) != 1: continue
                if get_bit(opp_occ, to_sq):
                    if r == 6:
                        for p in (W_QUEEN, W_ROOK, W_BISHOP, W_KNIGHT):
                            buf[n] = encode_move(sq, to_sq, p); n += 1
                    else:
                        buf[n] = encode_move(sq, to_sq); n += 1
                elif to_sq == ep_sq and ep_sq != 64:
                    buf[n] = encode_move(sq, to_sq); n += 1

        # ── White knights ────────────────────────────────────────────────────
        knights = get_pieces(board, W_KNIGHT)
        while knights:
            sq, knights = pop_lsb(knights)
            atk = knight_attacks_sq(sq) & ~my_occ
            while atk:
                to_sq, atk = pop_lsb(atk)
                buf[n] = encode_move(sq, to_sq); n += 1

        # ── White bishops ────────────────────────────────────────────────────
        bishops = get_pieces(board, W_BISHOP)
        while bishops:
            sq, bishops = pop_lsb(bishops)
            atk = bishop_attacks_sq(sq, occ) & ~my_occ
            while atk:
                to_sq, atk = pop_lsb(atk)
                buf[n] = encode_move(sq, to_sq); n += 1

        # ── White rooks ──────────────────────────────────────────────────────
        rooks = get_pieces(board, W_ROOK)
        while rooks:
            sq, rooks = pop_lsb(rooks)
            atk = rook_attacks_sq(sq, occ) & ~my_occ
            while atk:
                to_sq, atk = pop_lsb(atk)
                buf[n] = encode_move(sq, to_sq); n += 1

        # ── White queens ─────────────────────────────────────────────────────
        queens = get_pieces(board, W_QUEEN)
        while queens:
            sq, queens = pop_lsb(queens)
            atk = queen_attacks_sq(sq, occ) & ~my_occ
            while atk:
                to_sq, atk = pop_lsb(atk)
                buf[n] = encode_move(sq, to_sq); n += 1

        # ── White king ───────────────────────────────────────────────────────
        king_bb = get_pieces(board, W_KING)
        if king_bb:
            sq, _ = pop_lsb(king_bb)
            atk = king_attacks_sq(sq) & ~my_occ
            while atk:
                to_sq, atk = pop_lsb(atk)
                buf[n] = encode_move(sq, to_sq); n += 1
            # Castling
            if castling & int(CASTLE_WK):
                if not get_bit(occ,5) and not get_bit(occ,6):
                    if not is_square_attacked(board,4,1) and \
                       not is_square_attacked(board,5,1) and \
                       not is_square_attacked(board,6,1):
                        buf[n] = encode_move(4,6); n += 1
            if castling & int(CASTLE_WQ):
                if not get_bit(occ,3) and not get_bit(occ,2) and not get_bit(occ,1):
                    if not is_square_attacked(board,4,1) and \
                       not is_square_attacked(board,3,1) and \
                       not is_square_attacked(board,2,1):
                        buf[n] = encode_move(4,2); n += 1

    else:  # side == 1 (Black)
        # ── Black pawns ──────────────────────────────────────────────────────
        pawns = get_pieces(board, B_PAWN)
        while pawns:
            sq, pawns = pop_lsb(pawns)
            r = sq // 8
            if not get_bit(occ, sq - 8):
                if r == 1:   # promotion
                    for p in (B_QUEEN, B_ROOK, B_BISHOP, B_KNIGHT):
                        buf[n] = encode_move(sq, sq-8, p); n += 1
                else:
                    buf[n] = encode_move(sq, sq-8); n += 1
                    if r == 6 and not get_bit(occ, sq-16):
                        buf[n] = encode_move(sq, sq-16); n += 1
            for to_sq in (sq-7, sq-9):
                if to_sq < 0: continue
                tf = to_sq % 8
                if abs(tf - sq%8) != 1: continue
                if get_bit(opp_occ, to_sq):
                    if r == 1:
                        for p in (B_QUEEN, B_ROOK, B_BISHOP, B_KNIGHT):
                            buf[n] = encode_move(sq, to_sq, p); n += 1
                    else:
                        buf[n] = encode_move(sq, to_sq); n += 1
                elif to_sq == ep_sq and ep_sq != 64:
                    buf[n] = encode_move(sq, to_sq); n += 1

        # ── Black knights ────────────────────────────────────────────────────
        knights = get_pieces(board, B_KNIGHT)
        while knights:
            sq, knights = pop_lsb(knights)
            atk = knight_attacks_sq(sq) & ~my_occ
            while atk:
                to_sq, atk = pop_lsb(atk)
                buf[n] = encode_move(sq, to_sq); n += 1

        # ── Black bishops ────────────────────────────────────────────────────
        bishops = get_pieces(board, B_BISHOP)
        while bishops:
            sq, bishops = pop_lsb(bishops)
            atk = bishop_attacks_sq(sq, occ) & ~my_occ
            while atk:
                to_sq, atk = pop_lsb(atk)
                buf[n] = encode_move(sq, to_sq); n += 1

        # ── Black rooks ──────────────────────────────────────────────────────
        rooks = get_pieces(board, B_ROOK)
        while rooks:
            sq, rooks = pop_lsb(rooks)
            atk = rook_attacks_sq(sq, occ) & ~my_occ
            while atk:
                to_sq, atk = pop_lsb(atk)
                buf[n] = encode_move(sq, to_sq); n += 1

        # ── Black queens ─────────────────────────────────────────────────────
        queens = get_pieces(board, B_QUEEN)
        while queens:
            sq, queens = pop_lsb(queens)
            atk = queen_attacks_sq(sq, occ) & ~my_occ
            while atk:
                to_sq, atk = pop_lsb(atk)
                buf[n] = encode_move(sq, to_sq); n += 1

        # ── Black king ───────────────────────────────────────────────────────
        king_bb = get_pieces(board, B_KING)
        if king_bb:
            sq, _ = pop_lsb(king_bb)
            atk = king_attacks_sq(sq) & ~my_occ
            while atk:
                to_sq, atk = pop_lsb(atk)
                buf[n] = encode_move(sq, to_sq); n += 1
            if castling & int(CASTLE_BK):
                if not get_bit(occ,61) and not get_bit(occ,62):
                    if not is_square_attacked(board,60,0) and \
                       not is_square_attacked(board,61,0) and \
                       not is_square_attacked(board,62,0):
                        buf[n] = encode_move(60,62); n += 1
            if castling & int(CASTLE_BQ):
                if not get_bit(occ,59) and not get_bit(occ,58) and not get_bit(occ,57):
                    if not is_square_attacked(board,60,0) and \
                       not is_square_attacked(board,59,0) and \
                       not is_square_attacked(board,58,0):
                        buf[n] = encode_move(60,58); n += 1

    # ── Legality filter: remove moves that leave own king in check ───────────
    king_piece = W_KING if side == 0 else B_KING
    legal = np.zeros(n, dtype=np.int32)
    m = 0
    for i in range(n):
        nb = copy_board(board)
        make_move(nb, buf[i])
        king_bb2 = get_pieces(nb, king_piece)
        if king_bb2:
            ksq, _ = pop_lsb(king_bb2)
            if not is_square_attacked(nb, ksq, 1 - side):
                legal[m] = buf[i]; m += 1
    return legal[:m]

# =============================================================================
# ZOBRIST HASHING
# =============================================================================
np.random.seed(42)
ZOBRIST_PIECES  = np.random.randint(1, 2**63, size=(12, 64), dtype=np.int64).view(np.uint64)
ZOBRIST_SIDE    = np.uint64(np.random.randint(1, 2**63, dtype=np.int64))   # explicit np.uint64 scalar
ZOBRIST_CASTLE  = np.random.randint(1, 2**63, size=16, dtype=np.int64).view(np.uint64)
ZOBRIST_EP      = np.random.randint(1, 2**63, size=8,  dtype=np.int64).view(np.uint64)

@njit(cache=True)
def compute_hash(board, zp, zs, zc, ze):
    h = uint64(0)
    for sq in range(64):
        p = piece_on(board, sq)
        if p != -1:
            h ^= zp[p, sq]
    if get_side(board) == 1:
        h ^= zs
    h ^= zc[get_castling(board) & 15]
    ep = get_ep_square(board)
    if ep != 64:
        h ^= ze[ep % 8]
    return h

# =============================================================================
# TRANSPOSITION TABLE
# =============================================================================
FLAG_EXACT      = int32(1)
FLAG_LOWERBOUND = int32(2)
FLAG_UPPERBOUND = int32(3)

def create_tt():
    return Dict.empty(
        key_type=types.uint64,
        value_type=types.UniTuple(types.int32, 4),
    )

@njit(cache=True)
def tt_store(tt, h, depth, score, flag, best_move):
    key = uint64(h)
    if key in tt:
        old_depth, _, _, _ = tt[key]
        if old_depth > int32(depth):
            return
    tt[key] = (int32(depth), int32(score), int32(flag), int32(best_move))

@njit(cache=True)
def tt_probe(tt, h, depth, alpha, beta):
    key = uint64(h)
    if key not in tt:
        return False, int32(0), int32(0)
    stored_depth, stored_score, stored_flag, stored_move = tt[key]
    if stored_depth >= int32(depth):
        if stored_flag == FLAG_EXACT:
            return True, stored_score, stored_move
        if stored_flag == FLAG_LOWERBOUND and stored_score >= int32(beta):
            return True, stored_score, stored_move
        if stored_flag == FLAG_UPPERBOUND and stored_score <= int32(alpha):
            return True, stored_score, stored_move
    return False, int32(0), stored_move

# =============================================================================
# NEGAMAX
# =============================================================================

@njit(cache=True)
def negamax(board, depth, alpha, beta, tt, allow_null,
            zp, zs, zc, ze, l0w, l0b, l1w, l1_bias):
    """
    Negamax with alpha-beta pruning.
    evaluate() is already from side-to-move perspective — no * color needed.
    All weight arrays and Zobrist tables passed as arguments (cache-safe).
    """
    h = compute_hash(board, zp, zs, zc, ze)
    hit, tt_score, tt_move = tt_probe(tt, h, depth, alpha, beta)
    if hit:
        return tt_score

    if depth <= 0:
        return nnue_forward(board, l0w, l0b, l1w, l1_bias)

    # Null-move pruning
    side = get_side(board)
    if allow_null and depth >= 3:
        null_board = copy_board(board)
        make_null_move(null_board)
        null_score = -negamax(null_board, depth - 3, -beta, -beta + 1,
                              tt, False, zp, zs, zc, ze, l0w, l0b, l1w, l1_bias)
        if null_score >= beta:
            return beta

    moves = generate_legal_moves(board)
    if len(moves) == 0:
        # Check for checkmate vs stalemate
        king_piece = W_KING if side == 0 else B_KING
        king_bb = get_pieces(board, king_piece)
        if king_bb:
            ksq, _ = pop_lsb(king_bb)
            if is_square_attacked(board, ksq, 1 - side):
                return -INFINITY + 1   # checkmate
        return 0                       # stalemate

    # TT move first (move ordering)
    if tt_move != int32(0):
        for i in range(len(moves)):
            if moves[i] == tt_move:
                moves[0], moves[i] = moves[i], moves[0]
                break

    best_score  = -INFINITY
    best_move   = int32(0)
    orig_alpha  = alpha

    for i in range(len(moves)):
        move = moves[i]
        nb   = copy_board(board)
        make_move(nb, move)

        if i == 0:
            score = -negamax(nb, depth - 1, -beta, -alpha,
                             tt, True, zp, zs, zc, ze, l0w, l0b, l1w, l1_bias)
        else:
            # Zero-window search
            score = -negamax(nb, depth - 1, -alpha - 1, -alpha,
                             tt, True, zp, zs, zc, ze, l0w, l0b, l1w, l1_bias)
            if alpha < score < beta:
                score = -negamax(nb, depth - 1, -beta, -alpha,
                                 tt, True, zp, zs, zc, ze, l0w, l0b, l1w, l1_bias)

        if score > best_score:
            best_score = score
            best_move  = int32(move)

        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    flag = FLAG_EXACT
    if best_score <= orig_alpha:
        flag = FLAG_UPPERBOUND
    elif best_score >= beta:
        flag = FLAG_LOWERBOUND

    tt_store(tt, h, depth, best_score, flag, best_move)
    return best_score

# =============================================================================
# ITERATIVE DEEPENING + TIME MANAGEMENT
# =============================================================================

def get_best_move(board_array, time_left_ms,
                  l0w=None, l0b=None, l1w=None, l1_bias=None):
    """
    Iterative deepening search with time management.
    Weight arrays can be passed explicitly; if omitted the module globals
    (already loaded by initialize_nnue) are used.
    """
    if l0w    is None: l0w    = L0_WEIGHTS
    if l0b    is None: l0b    = L0_BIASES
    if l1w    is None: l1w    = L1_WEIGHTS
    if l1_bias is None: l1_bias = L1_BIAS

    # Snapshot Zobrist tables (avoids repeated global lookups in @njit)
    zp = ZOBRIST_PIECES
    zs = ZOBRIST_SIDE
    zc = ZOBRIST_CASTLE
    ze = ZOBRIST_EP

    # Fresh TT every call — no cross-game contamination
    tt = create_tt()

    # Time budget: ~1/30th of remaining time, clamped to [0.1s, 5s]
    budget_s = max(0.1, min(5.0, (time_left_ms / 1000.0) / 30.0))
    start    = time.time()

    def elapsed():
        return time.time() - start

    def time_ok(fraction=1.0):
        return elapsed() < budget_s * fraction

    last_best  = int32(0)
    last_score = 0

    for depth in range(1, 20):
        score = negamax(board_array, depth, -INFINITY, INFINITY,
                        tt, True, zp, zs, zc, ze, l0w, l0b, l1w, l1_bias)

        # Read best move from TT root entry
        h = compute_hash(board_array, zp, zs, zc, ze)
        key = uint64(h)
        if key in tt:
            _, _, _, mv = tt[key]
            if mv != int32(0):
                last_best = mv

        #print(f"  depth {depth:2d} | score {score:+7d} | move {int(last_best)} | t={elapsed():.2f}s")

        last_score = score

        # Stop if mate found
        if abs(score) >= INFINITY - 100:
            break

        # Don't start a new depth if we've used > 60% of budget
        if not time_ok(0.6):
            break

    return int(last_best)

# =============================================================================
# CACHE CLEARING — wipes stale Numba .nbi/.nbc files before each run
# =============================================================================

def _clear_numba_cache():
    """
    Delete any existing Numba __pycache__ entries for this package so that
    every process start compiles from scratch.

    WHY: Numba's @njit(cache=True) stores compiled machine code keyed by the
    argument *types* inferred at first-call time.  The Numba typed Dict used
    for the transposition table is a complex runtime object whose internal
    layout can differ between process restarts (e.g. after a draw/crash ends
    the previous game).  When the cached code is loaded but the Dict layout
    has shifted, Numba raises a low-level LLVM or segfault-style error during
    the warmup call — which is exactly the "crash on second run" symptom.

    Clearing the cache forces a fresh compilation on every startup.  The
    compilation takes ~25 s, which comfortably fits inside the platform's
    90 s init budget.  This matches the strategy used by the numba baseline,
    which never relies on a persistent cache at all (it uses python-chess and
    a simple @njit evaluate with no Dict).
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(src_dir):
        if os.path.basename(root) == "__pycache__":
            for fname in files:
                if fname.endswith((".nbi", ".nbc")):
                    try:
                        os.remove(os.path.join(root, fname))
                    except OSError:
                        pass


# =============================================================================
# JIT WARMUP — runs at import time, inside the 90s init budget
# =============================================================================

def _warmup():
    """
    Force Numba to JIT-compile every kernel in the call graph by running
    a depth-1 search on the starting position.  Called at module import
    time so compilation happens during the 90s init window, not on the
    first move clock.

    CHANGES vs original:
    - _clear_numba_cache() is called first so we always compile fresh,
      avoiding the "crash on second run" caused by stale .nbi/.nbc files
      that were written with a different Numba Dict internal layout.
    - The warmup is wrapped in a broad try/except so that any unexpected
      compilation error is reported but does not kill the agent process.
    - ZOBRIST_SIDE is now an explicit np.uint64 scalar (not a 0-d array
      extracted with [()]) so Numba's type inference is unambiguous and
      consistent across runs.
    """
    import time as _time
    from src.board import board_from_fen as _bff

    # ── Step 1: wipe stale cache so we always start clean ───────────────────
    _clear_numba_cache()

    t0 = _time.time()
    print("Warming up Numba JIT kernels...", flush=True)

    dummy = _bff("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    tt    = create_tt()

    # Zero-filled weight arrays — cache stores compiled code only, not data.
    # Types must exactly match what get_best_move() passes at runtime:
    #   l0w  : int16 (HIDDEN_SIZE, INPUT_SIZE)
    #   l0b  : int16 (HIDDEN_SIZE,)
    #   l1w  : int16 (2 * HIDDEN_SIZE,)
    #   l1b  : int32 scalar
    _l0w = np.zeros((128, 768), dtype=np.int16)
    _l0b = np.zeros(128,        dtype=np.int16)
    _l1w = np.zeros(256,        dtype=np.int16)
    _l1b = np.int32(0)

    # ── Step 2: compile the full call graph ─────────────────────────────────
    try:
        negamax(dummy, 1, -INFINITY, INFINITY,
                tt, False,
                ZOBRIST_PIECES, ZOBRIST_SIDE, ZOBRIST_CASTLE, ZOBRIST_EP,
                _l0w, _l0b, _l1w, _l1b)
        print(f"JIT warmup complete in {_time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        # If compilation still fails (e.g. Numba version mismatch), report it
        # clearly so it shows up in the validation log, but do NOT crash the
        # agent — the first real get_move() call will trigger recompilation.
        print(f"JIT warmup error (non-fatal, will recompile on first move): {e}",
              flush=True)


_warmup()
