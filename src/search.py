import time
import numpy as np
from numba import njit, uint64, int32, int8, uint8, uint16
from numba.typed import Dict
from numba.core import types

from src.board import copy_board, get_side, piece_on
# Import NNUE evaluation
from src.evaluate import evaluate

# =============================================================================
# STUBS FOR MEMBER 2 & 3
# =============================================================================
@njit(cache=True)
def generate_legal_moves(board):
    return np.empty(0, dtype=np.uint16) 

@njit(cache=True)
def make_move(board, move):
    pass

@njit(cache=True)
def make_null_move(board):
    # Member 3 will implement: flips the side to move and clears en passant
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
# TRANSPOSITION TABLE (Using Numba Typed Dict)
# =============================================================================
INFINITY = 999999

FLAG_NONE = 0
FLAG_EXACT = 1
FLAG_LOWERBOUND = 2
FLAG_UPPERBOUND = 3

# TT Entry: (depth, score, flag, best_move)
# We'll use a typed dict instead of parallel arrays to avoid readonly issues
def create_tt():
    """Create a new transposition table (called once at startup)"""
    tt = Dict.empty(
        key_type=types.uint64,
        value_type=types.UniTuple(types.int32, 4),  # (depth, score, flag, best_move)
    )
    return tt

# Global TT instance (created at module load)
TT = create_tt()

@njit(cache=True)
def tt_store(tt, h: np.uint64, depth: int, score: int, flag: int, best_move: int):
    """Store position in transposition table"""
    tt[h] = (int32(depth), int32(score), int32(flag), int32(best_move))

@njit(cache=True)
def tt_probe(tt, h: np.uint64, depth: int, alpha: int, beta: int):
    """
    Probe transposition table.
    Returns: (hit: bool, score: int32, move: int32)
    """
    if h not in tt:
        return False, int32(0), int32(0)
    
    stored_depth, stored_score, stored_flag, stored_move = tt[h]
    
    # Return move even if depth is insufficient
    if stored_depth < depth:
        return False, int32(0), int32(stored_move)
    
    # Check if we can use the stored score
    if stored_flag == FLAG_EXACT:
        return True, int32(stored_score), int32(stored_move)
    elif stored_flag == FLAG_LOWERBOUND and stored_score >= beta:
        return True, int32(stored_score), int32(stored_move)
    elif stored_flag == FLAG_UPPERBOUND and stored_score <= alpha:
        return True, int32(stored_score), int32(stored_move)
    
    return False, int32(0), int32(stored_move)

# =============================================================================
# NEGAMAX + ALPHA-BETA PRUNING
# =============================================================================
@njit(cache=True)
def negamax(board, depth, alpha, beta, color, tt, allow_null=True):
    h = compute_hash(board)
    hit, tt_score, tt_move = tt_probe(tt, h, depth, alpha, beta)
    if hit: 
        return tt_score

    if depth <= 0: 
        # NNUE evaluation returns score from side-to-move perspective
        # Multiply by color to convert to current search perspective
        return evaluate(board) * color

    # Null-Move Pruning (R=2 depth reduction)
    if allow_null and depth >= 3:
        null_board = copy_board(board)
        make_null_move(null_board)
        null_score = -negamax(null_board, depth - 3, -beta, -beta + 1, -color, tt, False)
        if null_score >= beta:
            return beta

    moves = generate_legal_moves(board)
    if len(moves) == 0: 
        return -INFINITY + 1 

    best_score = -INFINITY
    best_move = int32(0)
    original_alpha = alpha

    for i, move in enumerate(moves):
        new_board = copy_board(board)
        make_move(new_board, move)
        
        # Principal Variation Search (PVS)
        if i == 0:
            score = -negamax(new_board, depth - 1, -beta, -alpha, -color, tt, True)
        else:
            # Zero-window search
            score = -negamax(new_board, depth - 1, -alpha - 1, -alpha, -color, tt, True)
            if alpha < score < beta:
                # Re-search with full window if it fails high
                score = -negamax(new_board, depth - 1, -beta, -score, -color, tt, True)

        if score > best_score:
            best_score = score
            best_move = int32(move)

        alpha = max(alpha, score)
        if alpha >= beta: 
            break 

    flag = FLAG_EXACT
    if best_score <= original_alpha: 
        flag = FLAG_UPPERBOUND
    elif best_score >= beta: 
        flag = FLAG_LOWERBOUND
        
    tt_store(tt, h, depth, best_score, flag, best_move)
    return best_score

# =============================================================================
# ITERATIVE DEEPENING & TIME MANAGEMENT
# =============================================================================
def get_best_move(board_array, time_left_ms):
    """
    Find the best move using iterative deepening.
    
    Args:
        board_array: Current board position
        time_left_ms: Remaining time in milliseconds
    
    Returns:
        Best move (encoded as uint16)
    """
    start_time = time.time()
    base_time_limit = (time_left_ms * 0.03) / 1000.0 
    color = 1 if get_side(board_array) == 0 else -1
    
    last_completed_move = 0
    last_eval = 0
    
    # Use the global TT
    global TT
    
    for depth in range(1, 15): 
        # Pass TT to negamax
        score = negamax(board_array, depth, -INFINITY, INFINITY, color, TT, True)
        
        # Get best move from TT
        h = compute_hash(board_array)
        if h in TT:
            _, _, _, current_best = TT[h]
        else:
            current_best = last_completed_move
        
        # Extend time if the score swings dramatically (instability)
        time_limit = base_time_limit * 2.0 if abs(score - last_eval) > 150 else base_time_limit
        
        if time.time() - start_time > time_limit:
            if last_completed_move == 0: 
                last_completed_move = current_best
            break 
            
        last_completed_move = current_best
        last_eval = score
            
    return last_completed_move
