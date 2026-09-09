import struct
import sys
import os
import glob

# ── Locate checkpoint directory ───────────────────────────────────────────────
checkpoint_dir = os.path.join("src", "checkpoints", "1_simple")
if not os.path.isdir(checkpoint_dir):
    print(f"Directory not found: {checkpoint_dir}")
    sys.exit(1)

files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.bin")))
if not files:
    print("No .bin files found.")
    sys.exit(1)

print(f"Found {len(files)} .bin file(s):\n")

# ═══════════════════════════════════════════════════════════════════════════════
def try_formats(path):
    size = os.path.getsize(path)
    print(f"  File size : {size} bytes  ({size / 1024:.1f} KB)")

    with open(path, "rb") as f:
        raw = f.read(min(256, size))

    # ── Print first 16 int32s ─────────────────────────────────────────────────
    print("  First 16 × int32 (little-endian signed):")
    for i in range(0, min(64, len(raw) - 3), 4):
        v  = struct.unpack("<i", raw[i:i+4])[0]
        uv = struct.unpack("<I", raw[i:i+4])[0]
        f32 = struct.unpack("<f", raw[i:i+4])[0]
        print(f"    offset {i:3d}: i32={v:12d}  u32={uv:#010x}  f32={f32:+.6e}")

    print()

    # ── Brute-force int32 format ──────────────────────────────────────────────
    print("  [INT32] Brute-force (H in 32..2048, IN in 64..1024):")
    found_i32 = []
    for offset in range(0, min(64, len(raw) - 7), 4):
        H  = struct.unpack("<i", raw[offset:offset+4])[0]
        IN = struct.unpack("<i", raw[offset+4:offset+8])[0]
        if 32 <= H <= 2048 and 64 <= IN <= 1024:
            header_ints = offset // 4
            # With L1 bias
            exp = 4 * (header_ints + 2 + H * IN + H + 2 * H + 1)
            if exp == size:
                found_i32.append((offset, H, IN, "with_bias"))
                print(f"    MATCH (with bias)    offset={offset} H={H} IN={IN}")
            # Without L1 bias
            exp2 = 4 * (header_ints + 2 + H * IN + H + 2 * H)
            if exp2 == size:
                found_i32.append((offset, H, IN, "no_bias"))
                print(f"    MATCH (no bias)      offset={offset} H={H} IN={IN}")
            # With extra output bias vector (some formats have H output biases)
            exp3 = 4 * (header_ints + 2 + H * IN + H + 2 * H + H)
            if exp3 == size:
                found_i32.append((offset, H, IN, "extra_bias"))
                print(f"    MATCH (extra bias)   offset={offset} H={H} IN={IN}")
    if not found_i32:
        print("    No int32 match found.")

    # ── Brute-force float32 format ────────────────────────────────────────────
    print()
    print("  [FLOAT32] Brute-force (H in 32..2048, IN in 64..1024):")
    found_f32 = []
    for offset in range(0, min(64, len(raw) - 7), 4):
        H  = struct.unpack("<i", raw[offset:offset+4])[0]
        IN = struct.unpack("<i", raw[offset+4:offset+8])[0]
        if 32 <= H <= 2048 and 64 <= IN <= 1024:
            header_ints = offset // 4
            # float32 weights: same layout but floats instead of int32
            exp = 4 * (header_ints + 2 + H * IN + H + 2 * H + 1)
            if exp == size:
                found_f32.append((offset, H, IN, "with_bias"))
                print(f"    MATCH (with bias)    offset={offset} H={H} IN={IN}")
            exp2 = 4 * (header_ints + 2 + H * IN + H + 2 * H)
            if exp2 == size:
                found_f32.append((offset, H, IN, "no_bias"))
                print(f"    MATCH (no bias)      offset={offset} H={H} IN={IN}")
    if not found_f32:
        print("    No float32 match found.")

    # ── Try reading first plausible f32 values ────────────────────────────────
    print()
    print("  First 8 values as float32:")
    for i in range(0, min(32, len(raw) - 3), 4):
        v = struct.unpack("<f", raw[i:i+4])[0]
        print(f"    offset {i:3d}: {v:+.8f}")

    # ── Try reading as if it were a flat float32 weight dump ─────────────────
    # Some trainers dump: [L0_weights (H*IN floats), L0_bias (H floats),
    #                      L1_weights (2H floats), L1_bias (1 float)]
    # with NO header — just raw floats.
    print()
    print("  [NO-HEADER float32] Trying common architectures:")
    for H in [64, 128, 256]:
        for IN in [768, 640, 512]:
            exp_with    = 4 * (H * IN + H + 2 * H + 1)
            exp_without = 4 * (H * IN + H + 2 * H)
            if exp_with == size:
                print(f"    MATCH (no header, with bias)   H={H} IN={IN}")
            if exp_without == size:
                print(f"    MATCH (no header, no bias)     H={H} IN={IN}")

    # ── Try numpy-style .npy header ───────────────────────────────────────────
    if raw[:6] == b'\x93NUMPY':
        print()
        print("  *** This looks like a NumPy .npy file! ***")

    # ── Try PyTorch magic ─────────────────────────────────────────────────────
    PT_MAGIC = b'PK\x03\x04'   # zip header used by PyTorch
    if raw[:4] == PT_MAGIC:
        print()
        print("  *** This looks like a PyTorch .pt/.pth file (zip format)! ***")

    print()

# ═══════════════════════════════════════════════════════════════════════════════
for path in files:
    print("=" * 70)
    print(f"FILE: {os.path.basename(path)}")
    print("=" * 70)
    try_formats(path)
