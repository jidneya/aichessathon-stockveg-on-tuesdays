//! Minimal FEN -> Chess768 -> NNUE eval tool, built around bullet's
//! quantised inference format.
//!
//! Usage:
//!     cargo run --release -- "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
//!
//! Feature encoding was cross-checked against bullet's own data format
//! (jw1912/bulletformat, src/chess.rs) rather than assumed:
//!   - piece nibble = (colour << 3) | piece_kind
//!   - piece_kind:   0=Pawn 1=Knight 2=Bishop 3=Rook 4=Queen 5=King
//!   - square:       rank * 8 + file, a1 = 0 .. h8 = 63 (LERF)
//! Chess768's 768 = 2 colours * 6 piece kinds * 64 squares comes directly
//! from that same (colour, piece, square) triple, so the feature index is
//! `colour * 384 + piece_kind * 64 + square`, with the standard perspective-net
//! trick of mirroring the square (^ 56) and flipping colour for the
//! not-side-to-move accumulator.
//!
//! If your training run used a different piece ordering than the above,
//! everything downstream (PIECE_CHARS mapping in `parse_fen`) needs updating
//! to match, or you'll get a network that runs but evaluates nonsense.

// ---- Network hyperparameters (from your training config) ----
const HIDDEN_SIZE: usize = 128;
const QA: i16 = 255;
const QB: i16 = 64;
const SCALE: i32 = 400;

/// Loaded at compile time from the checkpoint bullet produced.
/// Path is relative to *this source file*.
static NNUE: Network = unsafe {
    std::mem::transmute(*include_bytes!("../checkpoints/quantised_2.bin"))
};

#[inline]
/// Square Clipped ReLU - Activation Function.
/// Range is 0.0 .. 1.0 (in other words, 0 to QA*QA quantized).
fn screlu(x: i16) -> i32 {
    let y = i32::from(x).clamp(0, i32::from(QA));
    y * y
}

/// This is the quantised format that bullet outputs.
#[repr(C)] // this means represent it like C does (fields in order)
pub struct Network {
    feature_weights: [Accumulator; 768],
    feature_bias: Accumulator,
    output_weights: [i16; 2 * HIDDEN_SIZE],
    output_bias: i16,
}

impl Network {
    /// Calculates the output of the network from the two perspective accumulators.
    pub fn evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        let mut output = 0;

        for (&input, &weight) in us.vals.iter().zip(&self.output_weights[..HIDDEN_SIZE]) {
            output += screlu(input) * i32::from(weight);
        }

        for (&input, &weight) in them.vals.iter().zip(&self.output_weights[HIDDEN_SIZE..]) {
            output += screlu(input) * i32::from(weight);
        }

        output /= i32::from(QA);
        output += i32::from(self.output_bias);
        output *= SCALE;
        output /= i32::from(QA) * i32::from(QB);

        output
    }
}

/// A column of the feature-weights matrix.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    vals: [i16; HIDDEN_SIZE],
}

impl Accumulator {
    pub fn new(net: &Network) -> Self {
        net.feature_bias
    }

    pub fn add_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self.vals.iter_mut().zip(&net.feature_weights[feature_idx].vals) {
            *i += *d
        }
    }

    #[allow(dead_code)]
    pub fn remove_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self.vals.iter_mut().zip(&net.feature_weights[feature_idx].vals) {
            *i -= *d
        }
    }
}

// ---------------------------------------------------------------------
// FEN parsing and Chess768 feature encoding
// ---------------------------------------------------------------------

/// One piece on the board, in absolute (non-perspective-flipped) terms.
struct Piece {
    /// 0 = white, 1 = black
    colour: usize,
    /// 0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king
    kind: usize,
    /// 0..64, a1 = 0, h8 = 63
    square: usize,
}

struct Position {
    pieces: Vec<Piece>,
    /// 0 = white to move, 1 = black to move
    stm: usize,
}

fn parse_fen(fen: &str) -> Result<Position, String> {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    let board_str = *parts.first().ok_or("empty FEN")?;
    let stm = match parts.get(1).copied() {
        Some("b") => 1,
        _ => 0, // treat missing/anything-else as white to move
    };

    const PIECE_CHARS: &str = "PNBRQKpnbrqk";

    let ranks: Vec<&str> = board_str.split('/').collect();
    if ranks.len() != 8 {
        return Err(format!("FEN board must have 8 ranks, found {}", ranks.len()));
    }

    let mut pieces = Vec::new();

    // FEN lists ranks from 8 down to 1; rank 1 is our rank index 0.
    for (i, rank_str) in ranks.iter().enumerate() {
        let rank = 7 - i;
        let mut file = 0usize;

        for ch in rank_str.chars() {
            if let Some(d) = ch.to_digit(10) {
                file += d as usize;
            } else {
                let idx = PIECE_CHARS
                    .find(ch)
                    .ok_or_else(|| format!("invalid piece character '{ch}'"))?;

                if file >= 8 {
                    return Err(format!("rank '{rank_str}' overflows 8 files"));
                }

                pieces.push(Piece {
                    colour: idx / 6,
                    kind: idx % 6,
                    square: rank * 8 + file,
                });
                file += 1;
            }
        }

        if file != 8 {
            return Err(format!("rank '{rank_str}' does not sum to 8 files"));
        }
    }

    Ok(Position { pieces, stm })
}

/// Builds the two perspective accumulators (us = side to move, them = other
/// side) for a position, using Chess768's feature layout:
/// `feature = colour * 384 + piece_kind * 64 + square`, mirroring the square
/// (^ 56) and flipping colour to get the opposite-perspective feature.
fn build_accumulators(net: &Network, pos: &Position) -> (Accumulator, Accumulator) {
    let mut us = Accumulator::new(net);
    let mut them = Accumulator::new(net);

    for p in &pos.pieces {
        let white_feat = p.colour * 384 + p.kind * 64 + p.square;
        let black_feat = (1 - p.colour) * 384 + p.kind * 64 + (p.square ^ 56);

        let (us_feat, them_feat) = if pos.stm == 0 {
            (white_feat, black_feat)
        } else {
            (black_feat, white_feat)
        };

        us.add_feature(us_feat, net);
        them.add_feature(them_feat, net);
    }

    (us, them)
}

/// Parses `fen`, builds accumulators, and runs the forward pass.
/// Returns the evaluation in centipawns, RELATIVE TO THE STM.
fn evaluate_fen(net: &Network, fen: &str) -> Result<i32, String> {
    let pos = parse_fen(fen)?;
    let (us, them) = build_accumulators(net, &pos);
    Ok(net.evaluate(&us, &them))
}

pub fn evaluate_fen_api(fen: &str) -> Result<i32, Box<dyn std::error::Error>>{
    match evaluate_fen(&NNUE, fen) {
        Ok(eval) => Ok(eval),
        Err(e) => Err(e.into()),
    }
}

// fn main() {
//     let args: Vec<String> = std::env::args().skip(1).collect();
//     if args.is_empty() {
//         eprintln!("Usage: nnue_eval \"<FEN>\"");
//         std::process::exit(1);
//     }
//     // Joined in case the shell splits an unquoted FEN on whitespace.
//     let fen = args.join(" ");

//     match evaluate_fen(&NNUE, &fen) {
//         Ok(eval) => println!("{eval}"),
//         Err(e) => {
//             eprintln!("Error: {e}");
//             std::process::exit(1);
//         }
//     }
// }