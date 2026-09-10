mod nnue;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> { // tis like Maybe in haskell
    // let fen = "8/p4k2/p1p5/8/1PpP1p2/2P3p1/1KR1q3/8 b -- 7 54";
    // nnue::evaluate_fen_api(fen);

    // will throw error if it doesn't work
    let mut reader = csv::Reader::from_path("../data/dataset_eval.csv")?;

    // to calculate averages:
    let mut differences: Vec<i32> = Vec::new();
    let mut count_too_bad = 0;
    let mut cnt_total = 0;

    for result in reader.records(){
        cnt_total += 1;
        let record = result?;
        let fen = &record[1];
        let black_to_move = fen.split_whitespace().nth(1) == Some("b"); //1 if black is to move
        
        if (record[3].starts_with('M')){
            continue;
        }
        //println!("eval field = {:?}", &record[3]);
        let mut actual_eval: i32 = record[3].parse()?;
        
        let mut nnue_eval = nnue::evaluate_fen_api(fen)?;
        //if (black_to_move) {nnue_eval = nnue_eval * -1;} // now it's in stm perspective, not white's

        let difference = (nnue_eval - actual_eval).abs();
        // if(difference > 200) {
        //     println!("{}, our eval: {}, actual eval: {}", fen, nnue_eval, actual_eval);
        // }
        if (difference > 500 && actual_eval.abs() < 500 || nnue_eval * actual_eval < 0){
            //println!("{}, our eval: {}, actual eval: {}", fen, nnue_eval, actual_eval);
            count_too_bad += 1;
        }

        differences.push(difference);
    }

    // Mean
    let mean = differences.iter().sum::<i32>() as f64 / differences.len() as f64;

    // Median
    differences.sort_unstable();

    let median = if differences.len() % 2 == 0 {
        let mid = differences.len() / 2;
        (differences[mid - 1] + differences[mid]) as f64 / 2.0
    } else {
        differences[differences.len() / 2] as f64
    };

    println!("Mean absolute difference:   {:.2}", mean);
    println!("Median absolute difference: {:.2}", median);
    println!("Number of bad / total : {} / {}", count_too_bad, cnt_total);

    Ok(())
}