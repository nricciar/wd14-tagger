use std::{process::exit};
use clap::Parser;

mod predictor;

use predictor::Predictor;

/// CLI to tag an image using ONNX model
#[derive(Parser, Debug)]
#[command(name = "ImageTagger")]
#[command(about = "W14 Image Tagger", long_about = None)]
struct Args {
    /// Path to the image file
    image: String,
}


fn main() {
    let args = Args::parse();

    let mut pred = Predictor::new(predictor::ModelKind::DINOv3);
    let img = match image::open(&args.image) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to open image '{}': {}", &args.image, e);
            exit(1);
        }
    };
    let output = pred.predict(&img, 0.35, false, 0.85, false, &[]);

    let g_str : Vec<String> = output.general.iter().map(|g| g.0.to_string()).collect();

    println!("{:?}", g_str);
}
