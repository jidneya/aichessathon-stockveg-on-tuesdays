import csv
import statistics
from nnue import Network, evaluate_fen

# ---------------------------------------------------------------------------
# Load network (equivalent to the static NNUE in Rust)
# ---------------------------------------------------------------------------

NET = Network.load("../checkpoints/quantised_2.bin")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    differences: list[int] = []
    count_too_bad: int = 0
    cnt_total: int = 0

    with open("../data/dataset_eval.csv", newline="", encoding="utf-8") as f:
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
            nnue_eval:   int = evaluate_fen(NET, fen)

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
    main()
