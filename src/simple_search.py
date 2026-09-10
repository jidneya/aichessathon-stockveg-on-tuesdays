import os
import time
import numpy as np
from numba import njit, uint64, int32
from numba.typed import Dict
from numba.core import types

from board import (
    board_from_fen, move_to_uci,
    copy_board, get_side, piece_on, encode_move, decode_move,
    get_pieces, get_white_occ, get_black_occ, get_all_occ,
    get_ep_square, get_castling, set_bit, clear_bit, get_bit,
    pop_lsb, lsb,
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ,
)

import nnue_fr as nnue

INFINITY = 999999
CONTEMPT = np.int32(-10)

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


import numpy as np
from numba import njit, uint64, int8, boolean

# ─────────────────────────────────────────────
#  NNUE — Accumulator builder
#  Chess768 feature layout:
#    feature = colour * 384 + kind * 64 + square
#  Mirror square (^ 56) and flip colour for the opposite perspective.
# ─────────────────────────────────────────────

# def build_accumulators(net: nnue.Network, pos: np.ndarray) -> tuple[nnue.Accumulator, nnue.Accumulator]:
#     """
#     Build both perspective accumulators from scratch for a given board state.

#     Args:
#         net : Network  — holds feature_weights and make_accumulator()
#         pos : np.ndarray (uint64, length 20) — the board state array

#     Returns:
#         (us, them) — accumulators for side-to-move and the other side,
#                      returned as raw int16 np.ndarrays (not Accumulator objects)
#     """
#     us   = net.make_accumulator()
#     them = net.make_accumulator()

#     stm = int(pos[17])   # 0 = white to move, 1 = black to move

#     for p in range(12):
#         colour = 0 if p < 6 else 1
#         kind   = p % 6

#         bb = pos[p]      # bitboard for this piece type
#         while bb:
#             sq, bb = pop_lsb(bb)

#             # Absolute (non-perspective) features
#             white_feat = colour * 384 + kind * 64 + sq
#             black_feat = (1 - colour) * 384 + kind * 64 + (sq ^ 56)

#             # Assign to the correct perspective based on side to move
#             if stm == 0:   # white to move → us = white perspective
#                 us_feat, them_feat = white_feat, black_feat
#             else:           # black to move → us = black perspective
#                 us_feat, them_feat = black_feat, white_feat

#             us.add_feature(us_feat, net)
#             them.add_feature(them_feat, net)

#     # Extract raw int16 arrays — callers work with these directly
#     return us.vals.copy(), them.vals.copy()


# # ─────────────────────────────────────────────
# #  NNUE — Feature index helper (njit)
# # ─────────────────────────────────────────────

# @njit(cache=True)
# def _make_features(piece_idx: int, sq: int, stm: int):
#     """
#     Compute (us_feat, them_feat) for a piece on a given square,
#     from the perspective of the node whose STM is `stm`.
#     All args are plain ints — fully njit compatible.
#     """
#     colour = 0 if piece_idx < 6 else 1
#     kind   = piece_idx % 6

#     white_feat = colour * 384 + kind * 64 + sq
#     black_feat = (1 - colour) * 384 + kind * 64 + (sq ^ 56)

#     if stm == 0:   # white to move → us = white perspective
#         return white_feat, black_feat
#     else:           # black to move → us = black perspective
#         return black_feat, white_feat


# # ─────────────────────────────────────────────
# #  NNUE — Incremental accumulator update (njit)
# #
# #  us, them      : raw int16 np.ndarrays, shape (HIDDEN_SIZE,)
# #  fw            : raw int16 np.ndarray,  shape (768, HIDDEN_SIZE)
# #                  extracted once from net.feature_weights before search
# #  pos           : board state BEFORE the move (uint64 array, length 20)
# #  move          : encoded move integer
# #
# #  Returns (child_us, child_them) with perspectives swapped,
# #  because after the move the STM flips.
# # ─────────────────────────────────────────────

# @njit(cache=True)
# def nnue_update_accumulators(us, them, fw, pos, move):
#     us_new   = us.copy()
#     them_new = them.copy()

#     stm = int(pos[17])
#     from_sq = move & 63
#     to_sq   = (move >> 6) & 63
#     promo   = (move >> 12) & 15

#     moving_piece   = piece_on(pos, from_sq)   # 0-11
#     captured_piece = piece_on(pos, to_sq)     # -1 if empty

#     # ── 1. Remove the moving piece from its origin ────────────────────────
#     us_feat, them_feat = _make_features(moving_piece, from_sq, stm)
#     us_new   -= fw[us_feat]
#     them_new -= fw[them_feat]

#     # ── 2. Remove any captured piece from the destination ─────────────────
#     if captured_piece != -1:
#         us_feat, them_feat = _make_features(captured_piece, to_sq, stm)
#         us_new   -= fw[us_feat]
#         them_new -= fw[them_feat]

#     # ── 3. En passant — remove the captured pawn from its real square ──────
#     ep_sq = int(pos[15])   # 64 = no en passant
#     is_ep = (
#         (moving_piece == W_PAWN or moving_piece == B_PAWN)
#         and to_sq == ep_sq
#         and ep_sq != 64
#     )
#     if is_ep:
#         ep_capture_sq = to_sq - 8 if stm == 0 else to_sq + 8
#         ep_pawn       = B_PAWN if stm == 0 else W_PAWN
#         us_feat, them_feat = _make_features(ep_pawn, ep_capture_sq, stm)
#         us_new   -= fw[us_feat]
#         them_new -= fw[them_feat]

#     # ── 4. Add the piece that lands on to_sq ──────────────────────────────
#     landing_piece = promo if promo != 0 else moving_piece
#     us_feat, them_feat = _make_features(landing_piece, to_sq, stm)
#     us_new   += fw[us_feat]
#     them_new += fw[them_feat]

#     # ── 5. Swap perspectives — child's STM is the opponent ────────────────
#     return them_new, us_new


# # ─────────────────────────────────────────────
# #  NNUE — Evaluation
# #  us, them are raw int16 arrays.
# #  net is still the full Network object — needed for output_weights/bias.
# # ─────────────────────────────────────────────

# def nnue_evaluate(us, them, net) -> int:
#     # Wrap back into Accumulator objects only for the evaluate call
#     us_acc   = nnue.Accumulator(us)
#     them_acc = nnue.Accumulator(them)
#     return net.evaluate(us_acc, them_acc)


# # ─────────────────────────────────────────────
# #  Search
# # ─────────────────────────────────────────────

# def minimax(net, fw, board, us, them, depth, maximising):
#     moves = generate_legal_moves(board)

#     if depth == 0 or not moves:
#         return nnue_evaluate(us, them, net), None

#     best_move  = None
#     best_score = -INFINITY if maximising else INFINITY

#     for move in moves:
#         child_board = copy_board(board)
#         make_move(child_board, move)
#         child_us, child_them = nnue_update_accumulators(us, them, fw, board, move)

#         score, _ = minimax(net, fw, child_board, child_us, child_them, depth - 1, not maximising)

#         if maximising and score > best_score:
#             best_score, best_move = score, move
#         elif not maximising and score < best_score:
#             best_score, best_move = score, move

#     return best_score, best_move


# def search(net, board, depth):
#     # Extract raw feature weight matrix once — passed as a plain ndarray into njit
#     fw = np.stack([acc.vals for acc in net.feature_weights], axis=0)  # (768, HIDDEN_SIZE) int16

#     us, them         = build_accumulators(net, board)   # raw int16 arrays
#     score, best_move = minimax(net, fw, board, us, them, depth, maximising=(int(board[17]) == 0))
#     print(f"Best move: {move_to_uci(best_move)} | Score: {score}")
#     return best_move


# if __name__ == "__main__":
#     net = nnue.load_network("../nnue_test/nnue_test/checkpoints/quantised_2.bin")
#     bd  = board_from_fen("r3r1k1/p2q1pp1/2p4p/8/1P1PRB2/P4Q1P/1P3PP1/4R1K1 b - - 4 20")
#     search(net, bd, 2)

# import numpy as np
# import nnue
# from board import pop_lsb, piece_on, W_PAWN, B_PAWN, generate_legal_moves, copy_board, make_move, board_from_fen, move_to_uci

INFINITY = 10_000_000


# ─────────────────────────────────────────────
#  Search
# ─────────────────────────────────────────────

def minimax(feature_weights, output_weights, output_bias, board, us, them, depth, alpha, beta, maximising):
    moves = generate_legal_moves(board)

    if depth == 0 or not moves:
        # nnue.evaluate takes raw int16 arrays + raw output weights/bias.
        # No wrapper objects, no re-boxing — fully compatible with the @njit path.
        score = nnue.evaluate(us, them, output_weights, output_bias)
        return score, None

    best_move  = None
    best_score = -INFINITY if maximising else INFINITY

    for move in moves:
        child_board = copy_board(board)
        make_move(child_board, move)

        # nnue.update_accumulators takes raw int16 accumulators + raw feature
        # weight matrix. Returns (child_us, child_them) already perspective-swapped.
        child_us, child_them = nnue.update_accumulators(
            us, them, feature_weights, board, move
        )

        score, _ = minimax(
            feature_weights, output_weights, output_bias,
            child_board, child_us, child_them,
            depth - 1, alpha, beta, not maximising
        )

        if maximising:
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)

        # Beta cutoff (maximiser) / Alpha cutoff (minimiser)
        if beta <= alpha:
            break

    return best_score, best_move


def search(net, board, depth):
    # net is a plain tuple — unpack it once here, then pass only what each
    # call site needs. No attribute access, no Python object overhead.
    feature_weights, feature_bias, output_weights, output_bias = net

    # Build the root accumulators from scratch using the @njit function.
    # feature_bias initialises both accumulators; feature_weights fills them.
    us, them = nnue.build_accumulators(feature_weights, feature_bias, board)

    score, best_move = minimax(
        feature_weights, output_weights, output_bias,
        board, us, them,
        depth,
        alpha=-INFINITY,
        beta=INFINITY,
        maximising=(int(board[17]) == 0)   # 0 = white to move
    )

    print(f"Best move: {move_to_uci(best_move)} | Score: {score}")
    return best_move


if __name__ == "__main__":
    # load_network returns a plain tuple — no class instantiation.
    net = nnue.load_network("../nnue_test/nnue_test/checkpoints/quantised_2.bin")
    # bd  = board_from_fen("r3r1k1/p2q1pp1/2p4p/8/1P1PRB2/P4Q1P/1P3PP1/4R1K1 b - - 4 20")
    # search(net, bd, 6)

    
