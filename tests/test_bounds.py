import numpy as np
from src.board import board_from_fen
from src.search import get_best_move

# Simulate an impossibly long game by passing 500 history hashes
dummy_history = [np.uint64(x) for x in range(500)]
board = board_from_fen("8/8/8/3k4/8/8/4K3/8 w - - 0 1")

# If the array bounds fix failed, this will instantly throw an IndexError.
# If successful, it will calculate safely and print the final statement.
get_best_move(board, 5000, dummy_history)
print("Search completed without out-of-bounds crash.")