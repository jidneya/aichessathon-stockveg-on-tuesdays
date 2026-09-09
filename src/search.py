import os
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
from src.evaluate import (
    nnue_forward, evaluate,
    PST_TABLE,
    QA, QB, SCALE,
)
import src.evaluate as ev  # Dynamic module reference


INFINITY = 999999
CONTEMPT = np.int32(-10)

# =============================================================================
# ZOBRIST HASHING  —  all arrays explicitly uint64
# =============================================================================
np.random.seed(42)
ZOBRIST_PIECES = np.random.randint(
    1, 2**63, size=(12, 64), dtype=np.int64
).view(np.uint64)

ZOBRIST_SIDE = np.uint64(
    np.random.randint(1, 2**63, dtype=np.int64)
)

ZOBRIST_CASTLE = np.random.randint(
    1, 2**63, size=(4,), dtype=np.int64
).view(np.uint64)

ZOBRIST_EP = np.random.randint(
    1, 2**63, size=(8,), dtype=np.int64
).view(np.uint64)

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

    board[piece] = clear_bit(board[piece], from_sq)
    if piece < 6:
        board[12] = clear_bit(board[12], from_sq)
    else:
        board[13] = clear_bit(board[13], from_sq)

    captured = piece_on(board, to_sq)
    if captured != -1:
        board[captured] = clear_bit(board[captured], to_sq)
        if captured < 6:
            board[12] = clear_bit(board[12], to_sq)
        else:
            board[13] = clear_bit(board[13], to_sq)

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

    if promo != 0:
        landing = promo
    else:
        landing = piece

    board[landing] = set_bit(board[landing], to_sq)
    if landing < 6:
        board[12] = set_bit(board[12], to_sq)
    else:
        board[13] = set_bit(board[13], to_sq)

    if piece == W_PAWN and to_sq - from_sq == 16:
        new_ep = from_sq + 8
    elif piece == B_PAWN and from_sq - to_sq == 16:
        new_ep = from_sq - 8

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

    board[15] = new_ep
    board[16] = castling
    board[14] = board[12] | board[13]   # FIX: keep ALL-occupancy bitboard in sync
    board[17] ^= 1   # FIX: flip side to move (was erroneously toggling board[14],
                     # the ALL-occupancy bitboard, instead of board[17])

# =============================================================================
# MOVE GENERATION
# =============================================================================
@njit(cache=True)
def generate_moves(board):
    moves = []
    side  = get_side(board)
    occ   = get_all_occ(board)
    my_occ  = get_white_occ(board) if side == 0 else get_black_occ(board)
    opp_occ = get_black_occ(board) if side == 0 else get_white_occ(board)

    if side == 0:
        pawns   = get_pieces(board, W_PAWN)
        knights = get_pieces(board, W_KNIGHT)
        bishops = get_pieces(board, W_BISHOP)
        rooks   = get_pieces(board, W_ROOK)
        queens  = get_pieces(board, W_QUEEN)
        kings   = get_pieces(board, W_KING)
    else:
        pawns   = get_pieces(board, B_PAWN)
        knights = get_pieces(board, B_KNIGHT)
        bishops = get_pieces(board, B_BISHOP)
        rooks   = get_pieces(board, B_ROOK)
        queens  = get_pieces(board, B_QUEEN)
        kings   = get_pieces(board, B_KING)

    ep_sq    = get_ep_square(board)
    castling = get_castling(board)

    # ── Pawns ─────────────────────────────────────────────────────────────────
    bb = pawns
    while bb:
        sq, bb = pop_lsb(bb)
        r, f = sq // 8, sq % 8
        if side == 0:
            # single push
            if r < 7 and not get_bit(occ, sq + 8):
                if r == 6:
                    for promo in [W_QUEEN, W_ROOK, W_BISHOP, W_KNIGHT]:
                        moves.append(encode_move(sq, sq+8, promo))
                else:
                    moves.append(encode_move(sq, sq+8, 0))
                # double push
                if r == 1 and not get_bit(occ, sq + 16):
                    moves.append(encode_move(sq, sq+16, 0))
            # captures
            for df in [-1, 1]:
                nf = f + df
                if 0 <= nf <= 7:
                    to = (r+1)*8 + nf
                    if get_bit(opp_occ, to):
                        if r == 6:
                            for promo in [W_QUEEN, W_ROOK, W_BISHOP, W_KNIGHT]:
                                moves.append(encode_move(sq, to, promo))
                        else:
                            moves.append(encode_move(sq, to, 0))
                    elif to == ep_sq and ep_sq != 64:
                        moves.append(encode_move(sq, to, 0))
        else:
            # single push
            if r > 0 and not get_bit(occ, sq - 8):
                if r == 1:
                    for promo in [B_QUEEN, B_ROOK, B_BISHOP, B_KNIGHT]:
                        moves.append(encode_move(sq, sq-8, promo))
                else:
                    moves.append(encode_move(sq, sq-8, 0))
                # double push
                if r == 6 and not get_bit(occ, sq - 16):
                    moves.append(encode_move(sq, sq-16, 0))
            # captures
            for df in [-1, 1]:
                nf = f + df
                if 0 <= nf <= 7:
                    to = (r-1)*8 + nf
                    if get_bit(opp_occ, to):
                        if r == 1:
                            for promo in [B_QUEEN, B_ROOK, B_BISHOP, B_KNIGHT]:
                                moves.append(encode_move(sq, to, promo))
                        else:
                            moves.append(encode_move(sq, to, 0))
                    elif to == ep_sq and ep_sq != 64:
                        moves.append(encode_move(sq, to, 0))

    # ── Knights ───────────────────────────────────────────────────────────────
    bb = knights
    while bb:
        sq, bb = pop_lsb(bb)
        atk = knight_attacks_sq(sq)
        atk &= ~my_occ
        while atk:
            to, atk = pop_lsb(atk)
            moves.append(encode_move(sq, to, 0))

    # ── Bishops ───────────────────────────────────────────────────────────────
    bb = bishops
    while bb:
        sq, bb = pop_lsb(bb)
        atk = bishop_attacks_sq(sq, occ) & ~my_occ
        while atk:
            to, atk = pop_lsb(atk)
            moves.append(encode_move(sq, to, 0))

    # ── Rooks ─────────────────────────────────────────────────────────────────
    bb = rooks
    while bb:
        sq, bb = pop_lsb(bb)
        atk = rook_attacks_sq(sq, occ) & ~my_occ
        while atk:
            to, atk = pop_lsb(atk)
            moves.append(encode_move(sq, to, 0))

    # ── Queens ────────────────────────────────────────────────────────────────
    bb = queens
    while bb:
        sq, bb = pop_lsb(bb)
        atk = queen_attacks_sq(sq, occ) & ~my_occ
        while atk:
            to, atk = pop_lsb(atk)
            moves.append(encode_move(sq, to, 0))

    # ── King ──────────────────────────────────────────────────────────────────
    bb = kings
    while bb:
        sq, bb = pop_lsb(bb)
        atk = king_attacks_sq(sq) & ~my_occ
        while atk:
            to, atk = pop_lsb(atk)
            moves.append(encode_move(sq, to, 0))

    # ── Castling ──────────────────────────────────────────────────────────────
    if side == 0:
        if castling & int(CASTLE_WK):
            if not get_bit(occ, 5) and not get_bit(occ, 6):
                if (not is_square_attacked(board, 4, 1) and
                    not is_square_attacked(board, 5, 1) and
                    not is_square_attacked(board, 6, 1)):
                    moves.append(encode_move(4, 6, 0))
        if castling & int(CASTLE_WQ):
            if not get_bit(occ, 1) and not get_bit(occ, 2) and not get_bit(occ, 3):
                if (not is_square_attacked(board, 4, 1) and
                    not is_square_attacked(board, 3, 1) and
                    not is_square_attacked(board, 2, 1)):
                    moves.append(encode_move(4, 2, 0))
    else:
        if castling & int(CASTLE_BK):
            if not get_bit(occ, 61) and not get_bit(occ, 62):
                if (not is_square_attacked(board, 60, 0) and
                    not is_square_attacked(board, 61, 0) and
                    not is_square_attacked(board, 62, 0)):
                    moves.append(encode_move(60, 62, 0))
        if castling & int(CASTLE_BQ):
            if not get_bit(occ, 57) and not get_bit(occ, 58) and not get_bit(occ, 59):
                if (not is_square_attacked(board, 60, 0) and
                    not is_square_attacked(board, 59, 0) and
                    not is_square_attacked(board, 58, 0)):
                    moves.append(encode_move(60, 58, 0))

    return moves

# =============================================================================
# LEGAL MOVE FILTER
# =============================================================================
@njit(cache=True)
def generate_legal_moves(board):
    pseudo = generate_moves(board)
    legal  = []
    side   = get_side(board)
    king_piece = W_KING if side == 0 else B_KING
    for move in pseudo:
        child = copy_board(board)
        make_move(child, move)
        king_bb = get_pieces(child, king_piece)
        if king_bb == uint64(0):
            continue
        king_sq = lsb(king_bb)
        if not is_square_attacked(child, king_sq, 1 - side):
            legal.append(move)
    return legal

# =============================================================================
# ZOBRIST HASH
# =============================================================================
@njit(cache=True)
def compute_hash(board, zp, zs, zc, ze):
    """Compute Zobrist hash — all operations strictly uint64."""
    h = uint64(0)
    for piece in range(12):
        bb = board[piece]
        while bb:
            sq, bb = pop_lsb(bb)
            h ^= zp[piece, sq]
    if get_side(board) == 1:
        h ^= zs
    castling = int(get_castling(board))
    if castling & 1: h ^= zc[0]
    if castling & 2: h ^= zc[1]
    if castling & 4: h ^= zc[2]
    if castling & 8: h ^= zc[3]
    ep = int(get_ep_square(board))
    if ep != 64:
        h ^= ze[ep % 8]
    return h   # guaranteed uint64

# =============================================================================
# REPETITION HELPER
# =============================================================================
@njit(cache=True)
def count_hash_in_history(h, hist, hist_len):
    """Count occurrences of uint64 hash h in hist[0:hist_len]."""
    count = int32(0)
    for i in range(hist_len):
        if hist[i] == h:
            count += int32(1)
    return count

# =============================================================================
# TRANSPOSITION TABLE HELPERS
# =============================================================================
TT_EXACT = np.int32(0)
TT_LOWER = np.int32(1)
TT_UPPER = np.int32(2)

def make_tt():
    """Create a fresh transposition table (uint64 → int64 packed entry)."""
    return Dict.empty(
        key_type=types.uint64,
        value_type=types.int64,
    )

@njit(cache=True)
def tt_pack(score: int32, depth: int32, flag: int32, move: int32) -> np.int64:
    """Pack TT entry into a single int64."""
    s = np.int64(score  & 0xFFFF)
    d = np.int64(depth  & 0xFF)
    f = np.int64(flag   & 0x3)
    m = np.int64(move   & 0xFFFF)
    return (m << np.int64(26)) | (f << np.int64(24)) | (d << np.int64(16)) | s

@njit(cache=True)
def tt_unpack(entry: np.int64):
    s = np.int32(entry & np.int64(0xFFFF))
    if s > np.int32(32767): s -= np.int32(65536)
    d = np.int32((entry >> np.int64(16)) & np.int64(0xFF))
    f = np.int32((entry >> np.int64(24)) & np.int64(0x3))
    m = np.int32((entry >> np.int64(26)) & np.int64(0xFFFF))
    return s, d, f, m

@njit(cache=True)
def quiescence(board, alpha, beta, ply, w0, b0, w1, b1, qa, qb, scale, pst):
    """Searches captures beyond depth 0 to resolve tactical instability."""
    stand_pat = evaluate(board, w0, b0, w1, b1, qa, qb, scale, pst)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    moves = generate_legal_moves(board)
    captures = []
    
    # Filter only captures
    for mv in moves:
        to_sq = (mv >> 6) & 63
        if piece_on(board, to_sq) != -1:
            captures.append(mv)

    # MVV-LVA for captures only
    scores = np.zeros(len(captures), dtype=np.int32)
    VALS = (100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000)
    for idx in range(len(captures)):
        mv = captures[idx]
        to_sq = (mv >> 6) & 63
        from_sq = mv & 63
        victim = piece_on(board, to_sq)
        attacker = piece_on(board, from_sq)
        scores[idx] = int32(VALS[victim] * 10 - VALS[attacker])

    # Insertion sort
    for i in range(1, len(captures)):
        key_move = captures[i]
        key_score = scores[i]
        j = i - 1
        while j >= 0 and scores[j] < key_score:
            captures[j + 1] = captures[j]
            scores[j + 1] = scores[j]
            j -= 1
        captures[j + 1] = key_move
        scores[j + 1] = key_score

    for mv in captures:
        child = copy_board(board)
        make_move(child, mv)
        score = -quiescence(child, -beta, -alpha, ply + int32(1), w0, b0, w1, b1, qa, qb, scale, pst)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha

# =============================================================================
# NEGAMAX WITH ALPHA-BETA, PVS, NULL-MOVE, REPETITION DETECTION
# =============================================================================
@njit(cache=True)
def negamax(
    board, depth, alpha, beta, ply,
    tt,
    zp, zs, zc, ze,
    w0, b0, w1, b1, qa, qb, scale,
    pst,
    hist,        # uint64 array, length 512
    hist_len,    # int32 — number of real game hashes in hist
):
    # ── Repetition detection (skip at root) ──────────────────────────────────
    h = compute_hash(board, zp, zs, zc, ze)

    if ply >= int32(1):
        if count_hash_in_history(h, hist, hist_len + ply) >= int32(1):
            return CONTEMPT

    # ── TT probe ─────────────────────────────────────────────────────────────
    tt_move = int32(0)
    if h in tt:
        entry = tt[h]
        tt_score, tt_depth, tt_flag, tt_move = tt_unpack(entry)
        if tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_score
            elif tt_flag == TT_LOWER:
                if tt_score > alpha:
                    alpha = tt_score
            elif tt_flag == TT_UPPER:
                if tt_score < beta:
                    beta = tt_score
            if alpha >= beta:
                return tt_score

# ── Terminal / leaf ───────────────────────────────────────────────────────
    moves = generate_moves(board)  # Fetch PSEUDO moves to save time

    if depth <= int32(0):
        # This replaces evaluate() with quiescence() to prevent opening blunder
        return quiescence(board, alpha, beta, ply, w0, b0, w1, b1, qa, qb, scale, pst)

    # ── Null-move pruning ─────────────────────────────────────────────────────
    if depth >= int32(3) and ply > int32(0):
        null_board = copy_board(board)
        null_board[17] ^= 1   
        null_board[15]  = 64  
        null_score = -negamax(
            null_board, depth - int32(3), -beta, -beta + int32(1), ply + int32(1),
            tt, zp, zs, zc, ze, w0, b0, w1, b1, qa, qb, scale, pst,
            hist, hist_len,
        )
        if null_score >= beta:
            return beta

    # ── Move ordering: TT move first, then captures ───────────────────────────
    ordered = []
    rest    = []
    for mv in moves:
        if mv == tt_move and tt_move != int32(0):
            ordered.append(mv)
        elif piece_on(board, (mv >> 6) & 63) != -1:
            ordered.append(mv)
        else:
            rest.append(mv)
    ordered.extend(rest)

    # ── Write current hash into hist scratch area ─────────────────────────────
    hist[hist_len + ply] = h

    best_score = int32(-INFINITY)
    best_move  = int32(0)
    flag       = TT_UPPER
    legal_played = 0

    for mv in ordered:
        child = copy_board(board)
        make_move(child, mv)

        # Inline legality check (prevents double-copying the board)
        side = get_side(board)
        king_piece = W_KING if side == 0 else B_KING
        king_bb = get_pieces(child, king_piece)
        if king_bb != uint64(0):
            king_sq = lsb(king_bb)
            if is_square_attacked(child, king_sq, 1 - side):
                continue  # Move was illegal, skip it

        if legal_played == 0:
            score = -negamax(
                child, depth - int32(1), -beta, -alpha, ply + int32(1),
                tt, zp, zs, zc, ze, w0, b0, w1, b1, qa, qb, scale, pst,
                hist, hist_len,
            )
        else:
            score = -negamax(
                child, depth - int32(1), -alpha - int32(1), -alpha, ply + int32(1),
                tt, zp, zs, zc, ze, w0, b0, w1, b1, qa, qb, scale, pst,
                hist, hist_len,
            )
            if alpha < score < beta:
                score = -negamax(
                    child, depth - int32(1), -beta, -alpha, ply + int32(1),
                    tt, zp, zs, zc, ze, w0, b0, w1, b1, qa, qb, scale, pst,
                    hist, hist_len,
                )

        legal_played += 1

        if score > best_score:
            best_score = score
            best_move  = mv

        if score > alpha:
            alpha = score
            flag  = TT_EXACT

        if alpha >= beta:
            flag = TT_LOWER
            break

    # If no moves were legal, it is checkmate or stalemate
    if legal_played == 0:
        side = get_side(board)
        king_piece = W_KING if side == 0 else B_KING
        king_bb = get_pieces(board, king_piece)
        if king_bb != uint64(0):
            king_sq = lsb(king_bb)
            if is_square_attacked(board, king_sq, 1 - side):
                return int32(-INFINITY + ply)
        return int32(0)
    
    # ── Store in TT ───────────────────────────────────────────────────────────
    tt[h] = tt_pack(best_score, depth, flag, best_move)

    return best_score


# =============================================================================
# ITERATIVE DEEPENING DRIVER
# =============================================================================
def get_best_move(board, time_left_ms: int, game_hist: list) -> int:
    """
    Iterative deepening search.
    game_hist: list of np.uint64 Zobrist hashes from the actual game so far.
    Returns the best move as a packed int32.
    """
    budget_s = max(0.1, min(5.0, time_left_ms / 30_000))
    start    = time.time()

    # Build history array — fixed size 512, first hist_len slots = game history
    hist     = np.zeros(512, dtype=np.uint64)
    hist_len = int32(min(len(game_hist), 256))
    for i in range(int(hist_len)):
        hist[i] = np.uint64(game_hist[i])

    tt = make_tt()

    last_best = int32(0)

    # Ensure at least one legal move exists
    legal = generate_legal_moves(board)
    if len(legal) == 0:
        return int32(0)
    last_best = int32(legal[0])

    for depth in range(1, 20):
        alpha = int32(-INFINITY)
        beta  = int32(INFINITY)

        best_this_depth = int32(last_best)
        best_score      = int32(-INFINITY)

        for mv in legal:
            child = copy_board(board)
            make_move(child, mv)

            score = -negamax(
                child, int32(depth - 1), -beta, -alpha, int32(1),
                tt,
                ZOBRIST_PIECES, ZOBRIST_SIDE, ZOBRIST_CASTLE, ZOBRIST_EP,
                ev.L0_WEIGHTS, ev.L0_BIASES, ev.L1_WEIGHTS, ev.L1_BIAS,
                QA, QB, SCALE,
                PST_TABLE,
                hist, hist_len,
            )

            if score > best_score:
                best_score      = score
                best_this_depth = int32(mv)

            if score > alpha:
                alpha = score

        last_best = best_this_depth

        elapsed = time.time() - start
        if elapsed > budget_s * 0.6:
            break

    return last_best


# =============================================================================
# NUMBA WARMUP  —  clears stale cache then compiles all kernels
# =============================================================================
def _clear_numba_cache():
    """Delete stale .nbi/.nbc files to prevent cross-run cache crashes."""
    for root, dirs, files in os.walk(os.path.dirname(__file__)):
        if os.path.basename(root) == "__pycache__":
            for fname in files:
                if fname.endswith((".nbi", ".nbc")):
                    try:
                        os.remove(os.path.join(root, fname))
                    except OSError:
                        pass


def warmup():
    print("Warming up Numba JIT kernels...")
    _clear_numba_cache()
    t0 = time.time()

    import src.evaluate as ev
    ev.initialize_nnue()
    
    from src.board import board_from_fen
    dummy = board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    # Dummy history — all zeros, uint64
    hist     = np.zeros(512, dtype=np.uint64)
    hist_len = int32(0)

    tt = make_tt()

    try:
        negamax(
            dummy, int32(1), int32(-INFINITY), int32(INFINITY), int32(0),
            tt,
            ZOBRIST_PIECES, ZOBRIST_SIDE, ZOBRIST_CASTLE, ZOBRIST_EP,
            ev.L0_WEIGHTS, ev.L0_BIASES, ev.L1_WEIGHTS, ev.L1_BIAS,
            QA, QB, SCALE,
            PST_TABLE,
            hist, hist_len,
        )
    except Exception as e:
        print(f"Warmup warning (non-fatal): {e}")

    print(f"JIT warmup complete in {time.time() - t0:.1f}s")
