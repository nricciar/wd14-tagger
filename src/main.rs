use clap::Parser;
use std::{fs::File, io::Write, process::exit};

mod predictor;

use predictor::{ModelKind, Predictor};

/// CLI to tag an image using ONNX model
#[derive(Parser, Debug)]
#[command(name = "ImageTagger")]
#[command(about = "W14 Image Tagger", long_about = None)]
struct Args {
    /// Path to the image file
    image: String,

    /// Model backend to use
    #[arg(short = 'm', long = "model", default_value = "wd14")]
    model: ModelKind,

    /// Optional output file to write results
    #[arg(short = 'o', long = "output")]
    output: Option<String>,
}

fn main() {
    let args = Args::parse();

    let mut pred = Predictor::new(args.model);
    let img = match image::open(&args.image) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to open image '{}': {}", &args.image, e);
            exit(1);
        }
    };
    let output = pred.predict(&img, 0.35, false, 0.85, false, &[]);

    let g_str: Vec<String> = output.general.iter().map(|g| g.0.to_string()).collect();
    let c_str: Vec<String> = output.characters.iter().map(|c| c.0.to_string()).collect();

    match args.output {
        Some(filename) => {
            if let Err(e) =
                File::create(&filename).and_then(|mut f| f.write_all(g_str.join(", ").as_bytes()))
            {
                eprintln!("Failed to write to {}: {}", filename, e);
                exit(1);
            }
        }
        None => {
            println!("Tags: {}", g_str.join(", "));
            println!("Rating: {:?}", output.rating);
            println!("Characters: {:?}", c_str.join(", "));
        }
    }
}
