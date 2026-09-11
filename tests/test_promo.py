from agent import _encoded_to_uci

# Source square f2 is 13, Target square f1 is 5, Promotion piece B_QUEEN is 10
# Move encoding format: [4 bits promo] [6 bits target] [6 bits source]
encoded_move = (10 << 12) | (5 << 6) | 13

uci_string = _encoded_to_uci(encoded_move)
print(f"Engine output: {uci_string}")