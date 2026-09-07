import threading
from board import(
    board_from_fen,
    move_to_uci,
    copy_board,
    #generate_legal_moves,  <-- Import your move generator function when built
)
from search import get_best_move, negamax

ponder_thread = None
stop_event = threading.Event()

def ponder_worker(fen):
    board_array = board_from_fen(fen)
    # Simple background search to populate TT while waiting
    depth = 1
    while not stop_event.is_set() and depth < 10:
        negamax(board_array, depth, -999999, 999999, 1, True)
        depth += 1

def get_move(fen: str, time_left_ms: int) -> str:
    global ponder_thread
    
    # 1. Stop any active pondering from the opponent's turn
    if ponder_thread and ponder_thread.is_alive():
        stop_event.set()
        ponder_thread.join()
        
    # 2. Calculate our best move
    board_array = board_from_fen(fen)
    best_encoded_move = get_best_move(board_array, time_left_ms)
    
    # 3. Start a new ponder thread for the predicted future state
    stop_event.clear()
    ponder_thread = threading.Thread(target=ponder_worker, args=(fen,), daemon=True)
    ponder_thread.start()

    return move_to_uci(best_encoded_move)