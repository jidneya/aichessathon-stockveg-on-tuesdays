import numpy as np
from numba import njit, int16, int32
import csv

from src.board import piece_on, get_side

# =============================================================================
# NNUE ARCHITECTURE CONSTANTS
# =============================================================================
INPUT_SIZE = 768  # 12 piece types × 64 squares
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1

# Quantization scales (from your training code)
QA = 255  # l0 weights and biases
QB = 64   # l1 weights
QAB = 255 * 64  # l1 biases

SCALE = 400  # eval_scale from training

# =============================================================================
# GLOBAL WEIGHT ARRAYS (loaded once at startup)
# =============================================================================
# L0_WEIGHTS stored in column-major format for efficient feature lookup
L0_WEIGHTS = np.zeros((HIDDEN_SIZE, INPUT_SIZE), dtype=np.int16)
L0_BIASES = np.zeros(HIDDEN_SIZE, dtype=np.int16)
L1_WEIGHTS = np.zeros(2 * HIDDEN_SIZE, dtype=np.int16)
L1_BIASES = np.int16(0)

# =============================================================================
# LOAD CHECKPOINT
# =============================================================================
def initialize_nnue(checkpoint_path="checkpoints/final_final_weights.bin"):
    """
    Load quantized NNUE weights from bullet checkpoint.
    
    File format (all little-endian):
    1. L0 weights: (128, 768) column-major i16 array
    2. L0 biases: (128,) i16 array
    3. L1 weights: (256,) i16 array
    4. L1 bias: (1,) i16 scalar
    """
    global L0_WEIGHTS, L0_BIASES, L1_WEIGHTS, L1_BIASES
    
    with open(checkpoint_path, 'rb') as f:
        # Read L0 weights: 128 × 768 = 98,304 values (column-major)
        l0w_flat = np.fromfile(f, dtype=np.int16, count=HIDDEN_SIZE * INPUT_SIZE)
        # Reshape: 768 columns of 128 rows each
        L0_WEIGHTS = l0w_flat.reshape((HIDDEN_SIZE, INPUT_SIZE), order='F')
        
        # Read L0 biases: 128 values
        L0_BIASES = np.fromfile(f, dtype=np.int16, count=HIDDEN_SIZE)
        
        # Read L1 weights: 256 values
        L1_WEIGHTS = np.fromfile(f, dtype=np.int16, count=2 * HIDDEN_SIZE)
        
        # Read L1 bias: 1 value
        L1_BIASES = np.fromfile(f, dtype=np.int16, count=1)[0]
    
    print(f" Loaded NNUE checkpoint from {checkpoint_path}")
    print(f"  L0 weights: {L0_WEIGHTS.shape}, L0 biases: {L0_BIASES.shape}")
    print(f"  L1 weights: {L1_WEIGHTS.shape}, L1 bias: scalar")

# =============================================================================
# FEATURE EXTRACTION (Chess768)
# =============================================================================
@njit(cache=True)
def get_feature_index(piece, square, perspective):
    """
    Compute Chess768 feature index for simple piece-square encoding.
    
    Args:
        piece: Piece type (0-11: WP,WN,WB,WR,WQ,WK,BP,BN,BB,BR,BQ,BK)
        square: Square index (0-63)
        perspective: 0 for white's view, 1 for black's view
    
    Returns:
        Feature index (0-767)
    """
    # Flip square for black's perspective (rank flip)
    if perspective == 1:
        square = square ^ 56  # Flip rank: XOR with 56
    
    # Convert piece to STM/NTM relative encoding
    if perspective == 0:  # White's perspective
        # White pieces (0-5) stay 0-5 (STM)
        # Black pieces (6-11) become 6-11 (NTM)
        piece_idx = piece
    else:  # Black's perspective
        # Black pieces (6-11) become 0-5 (STM)
        # White pieces (0-5) become 6-11 (NTM)
        if piece < 6:
            piece_idx = piece + 6
        else:
            piece_idx = piece - 6
    
    # Feature index: piece_type * 64 + square
    return piece_idx * 64 + square

@njit(cache=True)
def extract_active_features(board):
    """
    Extract active Chess768 features for both perspectives.
    
    Returns:
        stm_features: Active feature indices from side-to-move perspective
        ntm_features: Active feature indices from opponent perspective
    """
    side_to_move = get_side(board)
    
    # Collect features (max 32 pieces)
    stm_features = []
    ntm_features = []
    
    for sq in range(64):
        piece = piece_on(board, sq)
        if piece == -1:
            continue
        
        if side_to_move == 0:  # White to move
            stm_idx = get_feature_index(piece, sq, 0)  # White's perspective
            ntm_idx = get_feature_index(piece, sq, 1)  # Black's perspective
        else:  # Black to move
            stm_idx = get_feature_index(piece, sq, 1)  # Black's perspective
            ntm_idx = get_feature_index(piece, sq, 0)  # White's perspective
        
        stm_features.append(stm_idx)
        ntm_features.append(ntm_idx)
    
    return np.array(stm_features, dtype=np.int32), np.array(ntm_features, dtype=np.int32)

# =============================================================================
# NNUE FORWARD PASS
# =============================================================================
@njit(cache=True)
def screlu(x):
    """Squared Clipped ReLU activation: clamp(x, 0, QA)^2"""
    clamped = min(max(x, 0), QA)
    return clamped * clamped

@njit(cache=True)
def forward_l0(active_features, weights, biases):
    """
    Compute L0 layer output using sparse feature accumulation.
    
    Args:
        active_features: Array of active feature indices
        weights: L0 weight matrix (HIDDEN_SIZE, INPUT_SIZE)
        biases: L0 bias vector (HIDDEN_SIZE,)
    
    Returns:
        Hidden layer activations after SCReLU (HIDDEN_SIZE,)
    """
    # Start with biases (already quantized by QA)
    accumulator = biases.copy()
    
    # Accumulate weights for active features
    for feat_idx in active_features:
        for i in range(HIDDEN_SIZE):
            accumulator[i] += weights[i, feat_idx]
    
    # Apply SCReLU activation (outputs are in range [0, QA*QA])
    output = np.zeros(HIDDEN_SIZE, dtype=np.int32)
    for i in range(HIDDEN_SIZE):
        output[i] = screlu(accumulator[i])
    
    return output

@njit(cache=True)
def forward_l1(stm_hidden, ntm_hidden, weights, bias):
    """
    Compute L1 layer output (final evaluation).
    
    Args:
        stm_hidden: STM hidden layer after SCReLU (HIDDEN_SIZE,) - values in [0, QA*QA]
        ntm_hidden: NTM hidden layer after SCReLU (HIDDEN_SIZE,) - values in [0, QA*QA]
        weights: L1 weight vector (2*HIDDEN_SIZE,) - quantized by QB
        bias: L1 bias scalar - quantized by QA*QB
    
    Returns:
        Evaluation score in centipawns
    """
    # Compute weighted sum
    output = int32(0)
    
    # STM contribution
    for i in range(HIDDEN_SIZE):
        output += stm_hidden[i] * int32(weights[i])
    
    # NTM contribution
    for i in range(HIDDEN_SIZE):
        output += ntm_hidden[i] * int32(weights[HIDDEN_SIZE + i])
    
    # Reduce quantization: from QA*QA*QB to QA*QB
    output = output // QA
    
    # Add bias (quantized by QA*QB)
    output += int32(bias)
    
    # Apply eval scale
    output = output * SCALE
    
    # Remove quantization: divide by QA*QB
    output = output // (QA * QB)
    
    return int(output)

# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================
@njit(cache=True)
def evaluate(board):
    """
    Evaluate board position using NNUE.
    
    Args:
        board: Board state array
    
    Returns:
        Evaluation score in centipawns from side-to-move perspective
    """
    # Extract active features
    stm_features, ntm_features = extract_active_features(board)
    
    # Forward pass through L0 (feature transformation)
    stm_hidden = forward_l0(stm_features, L0_WEIGHTS, L0_BIASES)
    ntm_hidden = forward_l0(ntm_features, L0_WEIGHTS, L0_BIASES)
    
    # Forward pass through L1 (output layer)
    score = forward_l1(stm_hidden, ntm_hidden, L1_WEIGHTS, L1_BIASES)
    
    return score

# =============================================================================
# TESTING / DEBUGGING
# =============================================================================
def test_evaluation():
    # """Test the evaluation function on starting position."""
    from board import board_from_fen
    
    # # Starting position
    # board = board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # score = evaluate(board)
    # print(f"Starting position eval: {score} centipawns")
    
    # # Position after 1.e4
    # board = board_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    # score = evaluate(board)
    # print(f"After 1.e4 eval: {score} centipawns")
    differences: list[int] = []
    count_too_bad: int = 0
    cnt_total: int = 0

    with open("checkpoints/dataset_eval.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row (matches csv::Reader which skips headers by default)

        for record in reader:
            cnt_total += 1

            fen         = record[1]
            eval_field  = record[3]

            # Skip mate scores (start with 'M'), mirroring the Rust continue
            if eval_field.startswith("M"):
                continue

            actual_eval: int = int(eval_field)
            nnue_eval:   int = evaluate(board_from_fen(fen))

            difference = abs(nnue_eval - actual_eval)

            # Mirror Rust's "too bad" condition exactly:
            # (difference > 500 && actual_eval.abs() < 500 || nnue_eval * actual_eval < 0)
            if (difference > 500 and abs(actual_eval) < 500) or (nnue_eval * actual_eval < 0):
                count_too_bad += 1

            differences.append(difference)

            if (cnt_total % 1000 == 0):
                print (cnt_total)
            

    # ---- Statistics --------------------------------------------------------

    # Mean
    mean = sum(differences) / len(differences)

    # Median  (mirrors Rust's manual even/odd split)
    differences.sort()
    n = len(differences)
    if n % 2 == 0:
        mid = n // 2
        median = (differences[mid - 1] + differences[mid]) / 2.0
    else:
        median = float(differences[n // 2])

    print(f"Mean absolute difference:   {mean:.2f}")
    print(f"Median absolute difference: {median:.2f}")
    print(f"Number of bad / total : {count_too_bad} / {cnt_total}")
    

if __name__ == "__main__":
    initialize_nnue("checkpoints/final_final_weights.bin")
    test_evaluation()
