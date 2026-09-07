import time
import numpy as np
from numba import njit, uint64, int32, int8, uint8, uint16

from board import copy_board, get_side, piece_on
# Import NNUE evaluation
from evaluate import evaluate

# =============================================================================
# STUBS FOR MEMBER 2 & 3
# =============================================================================
@njit(cache=True)
def generate_legal_moves(board):
    return np.empty(0, dtype=np.uint16) 

@njit(cache=True)
def make_move(board, move):
    pass

# =============================================================================
# ZOBRIST HASHING
# =============================================================================
np.random.seed(42)
ZOBRIST_PIECES = np.random.randint(0, 9223372036854775807, size=(12, 64), dtype=np.uint64)
ZOBRIST_SIDE = np.random.randint(0, 9223372036854775807, dtype=np.uint64)

@njit(cache=True)
def compute_hash(board: np.ndarray) -> np.uint64:
    h = np.uint64(0)
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
TT_SIZE = 1048576 
TT_MASK = TT_SIZE - 1

TT_HASH = np.zeros(TT_SIZE, dtype=np.uint64)
TT_SCORE = np.zeros(TT_SIZE, dtype=np.int32)
TT_DEPTH = np.zeros(TT_SIZE, dtype=np.int8)
TT_FLAG = np.zeros(TT_SIZE, dtype=np.uint8)
TT_MOVE = np.zeros(TT_SIZE, dtype=np.uint16)

FLAG_NONE = 0
FLAG_EXACT = 1
FLAG_LOWERBOUND = 2
FLAG_UPPERBOUND = 3
INFINITY = 999999

@njit(cache=True)
def tt_store(h: np.uint64, depth: int, score: int, flag: int, best_move: int):
    index = int(h & TT_MASK)
    TT_HASH[index] = h
    TT_DEPTH[index] = depth
    TT_SCORE[index] = score
    TT_FLAG[index] = flag
    TT_MOVE[index] = best_move

@njit(cache=True)
def tt_probe(h: np.uint64, depth: int, alpha: int, beta: int):
    index = int(h & TT_MASK)
    hit = False
    score = int32(0)
    move = np.uint16(0)
    
    if TT_HASH[index] == h:
        move = TT_MOVE[index]
        if TT_DEPTH[index] >= depth:
            flag = TT_FLAG[index]
            s = TT_SCORE[index]
            if flag == FLAG_EXACT:
                hit = True; score = s
            elif flag == FLAG_LOWERBOUND and s >= beta:
                hit = True; score = s
            elif flag == FLAG_UPPERBOUND and s <= alpha:
                hit = True; score = s
                
    # Numba demands uniform return types: (boolean, int32, uint16)
    return hit, score, move

# =============================================================================
# NEGAMAX + ALPHA-BETA PRUNING
# =============================================================================
@njit(cache=True)
def make_null_move(board):
    # Member 3 will implement: flips the side to move and clears en passant
    pass

@njit(cache=True)
def negamax(board, depth, alpha, beta, color, allow_null=True):
    h = compute_hash(board)
    hit, tt_score, tt_move = tt_probe(h, depth, alpha, beta)
    if hit: return tt_score

    if depth <= 0: 
        # NNUE evaluation returns score from side-to-move perspective
        # Multiply by color to convert to current search perspective
        return evaluate(board) * color

    # Null-Move Pruning (R=2 depth reduction)
    if allow_null and depth >= 3:
        null_board = copy_board(board)
        make_null_move(null_board)
        null_score = -negamax(null_board, depth - 3, -beta, -beta + 1, -color, False)
        if null_score >= beta:
            return beta

    moves = generate_legal_moves(board)
    if len(moves) == 0: return -INFINITY + 1 

    best_score = -INFINITY
    best_move = np.uint16(0)
    original_alpha = alpha

    for i, move in enumerate(moves):
        new_board = copy_board(board)
        make_move(new_board, move)
        
        # Principal Variation Search (PVS)
        if i == 0:
            score = -negamax(new_board, depth - 1, -beta, -alpha, -color, True)
        else:
            # Zero-window search
            score = -negamax(new_board, depth - 1, -alpha - 1, -alpha, -color, True)
            if alpha < score < beta:
                # Re-search with full window if it fails high
                score = -negamax(new_board, depth - 1, -beta, -score, -color, True)

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)
        if alpha >= beta: break 

    flag = FLAG_EXACT
    if best_score <= original_alpha: flag = FLAG_UPPERBOUND
    elif best_score >= beta: flag = FLAG_LOWERBOUND
        
    tt_store(h, depth, best_score, flag, best_move)
    return best_score

# =============================================================================
# ITERATIVE DEEPENING & TIME MANAGEMENT
# =============================================================================
def get_best_move(board_array, time_left_ms):
    start_time = time.time()
    base_time_limit = (time_left_ms * 0.03) / 1000.0 
    color = 1 if get_side(board_array) == 0 else -1
    
    last_completed_move = 0
    last_eval = 0
    
    for depth in range(1, 15): 
        # Pass True for allow_null on the initial call
        score = negamax(board_array, depth, -INFINITY, INFINITY, color, True)
        
        h = compute_hash(board_array)
        index = int(h & TT_MASK)
        current_best = TT_MOVE[index]
        
        # Extend time if the score swings dramatically (instability)
        time_limit = base_time_limit * 2.0 if abs(score - last_eval) > 150 else base_time_limit
        
        if time.time() - start_time > time_limit:
            if last_completed_move == 0: last_completed_move = current_best
            break 
            
        last_completed_move = current_best
        last_eval = score
            
    return last_completed_move
