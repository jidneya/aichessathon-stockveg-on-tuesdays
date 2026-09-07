import threading
from src.board import (
    board_from_fen,
    move_to_uci,
    copy_board,
    # generate_legal_moves,  <-- Import your move generator function when built
)
from src.search import get_best_move, negamax
from src.evaluate import initialize_nnue

# =============================================================================
# NNUE INITIALIZATION (CALLED ONCE AT MODULE LOAD)
# =============================================================================
print("Initializing NNUE evaluation...")
initialize_nnue("src/checkpoints/1_simple/quantised.bin")
print("NNUE ready!\n")

# =============================================================================
# PONDERING (BACKGROUND SEARCH)
# =============================================================================
ponder_thread = None
stop_event = threading.Event()

def ponder_worker(fen):
    """
    Background search while waiting for opponent's move.
    Populates transposition table with useful entries.
    """
    board_array = board_from_fen(fen)
    depth = 1
    
    # Iteratively deepen until stopped
    while not stop_event.is_set() and depth < 10:
        try:
            negamax(board_array, depth, -999999, 999999, 1, True)
            depth += 1
        except Exception as e:
            print(f"Pondering error at depth {depth}: {e}")
            break

# =============================================================================
# MAIN MOVE SELECTION FUNCTION
# =============================================================================
def get_move(fen: str, time_left_ms: int) -> str:
    """
    Main entry point called by the game server.
    
    Args:
        fen: Current board position in FEN notation
        time_left_ms: Remaining time in milliseconds
    
    Returns:
        Best move in UCI format (e.g., "e2e4")
    """
    global ponder_thread
    
    # 1. Stop any active pondering from the opponent's turn
    if ponder_thread and ponder_thread.is_alive():
        stop_event.set()
        ponder_thread.join(timeout=0.1)  # Wait up to 100ms
        
    # 2. Parse the board position
    board_array = board_from_fen(fen)
    
    # 3. Search for the best move
    print(f"Searching position: {fen[:50]}...")
    print(f"Time remaining: {time_left_ms}ms")
    
    best_encoded_move = get_best_move(board_array, time_left_ms)
    
    if best_encoded_move == 0:
        print("WARNING: No move found! Returning null move.")
        return "0000"  # Null move (should never happen with legal moves)
    
    # 4. Convert to UCI format
    uci_move = move_to_uci(best_encoded_move)
    print(f"Selected move: {uci_move}\n")
    
    # 5. Start pondering for the next position (optional)
    # Uncomment if you want background thinking during opponent's turn
    # stop_event.clear()
    # ponder_thread = threading.Thread(target=ponder_worker, args=(fen,), daemon=True)
    # ponder_thread.start()

    return uci_move

# =============================================================================
# TESTING / DEBUGGING
# =============================================================================
if __name__ == "__main__":
    """
    Test the agent with some sample positions.
    """
    print("=" * 60)
    print("TESTING AGENT")
    print("=" * 60)
    
    # Test 1: Starting position
    print("\nTest 1: Starting position")
    fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move1 = get_move(fen1, 60000)  # 60 seconds
    print(f"Result: {move1}")
    
    # Test 2: After 1.e4
    print("\nTest 2: After 1.e4")
    fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    move2 = get_move(fen2, 60000)
    print(f"Result: {move2}")
    
    # Test 3: Tactical position (scholar's mate setup)
    print("\nTest 3: Tactical position")
    fen3 = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    move3 = get_move(fen3, 60000)
    print(f"Result: {move3}")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
