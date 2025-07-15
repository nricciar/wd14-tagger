use ort::{Environment, SessionBuilder, Value};
use std::{fs::File, io::Write, process::exit};
use csv::{ReaderBuilder};
use serde::Deserialize;
use ndarray::Array4;
use image::{DynamicImage, RgbaImage};
use std::sync::Arc;
use image::GenericImage; // Add this
use ndarray::CowArray;
use ort::tensor::OrtOwnedTensor;
use ndarray::Ix2;
use clap::Parser;
use hf_hub::api::sync::Api;

/// CLI to tag an image using ONNX model
#[derive(Parser, Debug)]
#[command(name = "ImageTagger")]
#[command(about = "W14 Image Tagger", long_about = None)]
struct Args {
    /// Path to the image file
    image: String,

    /// Optional output file to write results
    #[arg(short = 'o', long = "output")]
    output: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TagRow {
    name: String,
    category: u8,
}

fn load_labels() -> (Vec<String>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let api = Api::new().unwrap();
    let repo = api.model("SmilingWolf/wd-vit-large-tagger-v3".to_string());
    let tag_filename = repo.get("selected_tags.csv").unwrap();
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(tag_filename).unwrap();
    let mut names = Vec::new();
    let mut ratings = Vec::new();
    let mut general = Vec::new();
    let mut character = Vec::new();
    for (idx, res) in rdr.deserialize().enumerate() {
        let row: TagRow = res.unwrap();
        let name = if row.name.chars().all(|c| "_()<>+^.0123456789".contains(c)) {
            row.name.clone()
        } else {
            row.name.replace('_', " ")
        };
        match row.category {
            9 => ratings.push(idx),
            0 => general.push(idx),
            4 => character.push(idx),
            _ => {}
        }
        names.push(name);
    }
    (names, ratings, general, character)
}

fn mcut_threshold(probs: &mut [f32]) -> f32 {
    probs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let diffs: Vec<_> = probs.windows(2).map(|w| w[0] - w[1]).collect();
    let t = diffs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    (probs[t] + probs[t + 1]) / 2.0
}

struct Predictor {
    session: Option<ort::Session>,
    size: usize,
    tag_names: Vec<String>,
    rating_i: Vec<usize>,
    general_i: Vec<usize>,
    character_i: Vec<usize>,
}

impl Predictor {
    fn new() -> Self {
        Predictor {
            session: None,
            size: 0,
            tag_names: vec![],
            rating_i: vec![],
            general_i: vec![],
            character_i: vec![],
        }
    }

    fn load(&mut self) {
        let api = Api::new().unwrap();
        let repo = api.model("SmilingWolf/wd-vit-large-tagger-v3".to_string());
        let model_filename = repo.get("model.onnx").unwrap();

        if self.session.is_some() { return; }
        let (tags, r, g, c) = load_labels();
        self.tag_names = tags;
        self.rating_i = r;
        self.general_i = g;
        self.character_i = c;

        let environment = Arc::new(Environment::builder().with_name("wd").build().unwrap());
        let session = SessionBuilder::new(&environment)
            .unwrap()
            .with_model_from_file(model_filename)
            .unwrap();
        println!("Input shape: {:?}", session.inputs[0].dimensions);
        let input_shape = session.inputs[0].dimensions.as_slice().to_vec();
        self.size = input_shape[2].expect("dimension is missing") as usize;

        self.session = Some(session);
    }

    fn prepare(&self, img: DynamicImage) -> Array4<f32> {
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let m = w.max(h);
        let mut canvas = RgbaImage::new(m, m);
        canvas.copy_from(&rgba, (m - w)/2, (m - h)/2).unwrap();
        let resized = image::imageops::resize(&canvas, self.size as u32, self.size as u32, image::imageops::FilterType::CatmullRom);
        let rgb = DynamicImage::ImageRgba8(resized).to_rgb8();

        let mut arr = Array4::<f32>::zeros((1, self.size, self.size, 3));
        for y in 0..self.size {
            for x in 0..self.size {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    arr[(0, y, x, c)] = pixel[2 - c] as f32;
                }
            }
        }
        arr
    }

    fn predict(
        &mut self,
        img: DynamicImage,
        g_th: f32,
        g_mcut: bool,
        c_th: f32,
        c_mcut: bool,
    ) -> (String, Vec<(String, f32)>, Vec<(String, f32)>, Vec<(String, f32)>) {
        self.load();
        let arr = self.prepare(img);
        let session = self.session.as_ref().unwrap();

        let arr_cow = CowArray::from(arr.into_dyn());
        let input = Value::from_array(session.allocator(), &arr_cow).unwrap();
        let outputs = session.run(vec![input]).unwrap();
        let preds: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
        let preds = preds.view().to_owned().into_dimensionality::<Ix2>().unwrap();
        let scores = preds.row(0);

        let rating = self.rating_i.iter()
            .map(|&i| (self.tag_names[i].clone(), scores[i]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let mut general = self.general_i.iter().map(|&i| (self.tag_names[i].clone(), scores[i])).collect::<Vec<_>>();
        let mut character = self.character_i.iter().map(|&i| (self.tag_names[i].clone(), scores[i])).collect::<Vec<_>>();

        if g_mcut {
            let mut gp: Vec<f32> = general.iter().map(|(_, v)| *v).collect();
            let thresh = mcut_threshold(&mut gp);
            general.retain(|&(_, v)| v > thresh);
        } else {
            general.retain(|&(_, v)| v > g_th);
        }

        if c_mcut {
            let mut cp: Vec<f32> = character.iter().map(|(_, v)| *v).collect();
            let thresh = mcut_threshold(&mut cp).max(0.15);
            character.retain(|&(_, v)| v > thresh);
        } else {
            character.retain(|&(_, v)| v > c_th);
        }

        general.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let general_str = general.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>().join(", ");

        (general_str, vec![rating], character, general)
    }
}

fn main() {
    let args = Args::parse();

    let mut pred = Predictor::new();
    let img = match image::open(&args.image) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to open image '{}': {}", &args.image, e);
            exit(1);
        }
    };
    let (g_str, rating, char_res, _gen_res) = pred.predict(img, 0.35, false, 0.85, false);

    match args.output {
        Some(filename) => {
            if let Err(e) = File::create(&filename).and_then(|mut f| f.write_all(g_str.as_bytes())) {
                eprintln!("Failed to write to {}: {}", filename, e);
                exit(1);
            }
        }
        None => {
            println!("Tags: {}", g_str);
            println!("Rating: {:?}", rating);
            println!("Characters: {:?}", char_res);
        }
    }
    
}
