use clap::ValueEnum;
use csv::ReaderBuilder;
use hf_hub::api::sync::Api;
use image::{DynamicImage, GenericImage, RgbaImage};
use ndarray::Array4;
use ndarray::Ix2;
use ort::{ep, session::Session, value::Tensor};
use serde::{Deserialize, Serialize};

// ── WD14 constants ────────────────────────────────────────────────────────────
pub const WD14_REPO: &str = "deepghs/wd14_tagger_with_embeddings";
pub const WD14_MODEL_FILE: &str = "SmilingWolf/wd-vit-large-tagger-v3/model.onnx";
pub const WD14_TAG_CSV: &str = "SmilingWolf/wd-vit-large-tagger-v3/tags_info.csv";

// ── DINOv3 constants ──────────────────────────────────────────────────────────
// ONNX weights live in the silveroxides conversion repo.
pub const DINO_ONNX_REPO: &str = "silveroxides/tagger-experiment-onnx";
pub const DINO_TAGGER_MODEL: &str = "tagger/model.onnx";
pub const DINO_TAGGER_DATA: &str = "tagger/model.onnx.data"; // must be beside model.onnx
pub const DINO_EMBED_MODEL: &str = "embedding/model.onnx";
pub const DINO_EMBED_DATA: &str = "embedding/model.onnx.data";
// Vocabulary lives in the original (non-ONNX) base repo.
pub const DINO_BASE_REPO: &str = "lodestones/tagger-experiment";
pub const DINO_VOCAB_FILE: &str = "tagger_vocab_with_categories.json";

// ImageNet normalisation for DINOv3 preprocessing
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const DINO_PATCH_SIZE: u32 = 16;
const DINO_MAX_SIZE: u32 = 1024;

// ── Public model selector ─────────────────────────────────────────────────────

#[derive(Serialize, Debug, Deserialize, Clone)]
pub struct OutputData {
    pub general: Vec<(String, f32)>,
    pub rating: (String, f32),
    pub characters: Vec<(String, f32)>,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
pub enum ModelKind {
    #[value(name = "wd14")]
    Wd14,
    #[value(name = "dino")]
    DINOv3,
}

// ── Vocabulary / label types ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Wd14TagRow {
    name: String,
    category: u8,
}

/// Expected shape of `tagger_vocab_with_categories.json`:
///   { "idx2tag": ["tag", ...], "idx2cat": [0, 9, 4, ...] }
///
/// Category codes mirror WD14: 0 = general, 4 = character, 9 = rating.
/// If `idx2cat` is absent (plain vocab file), every tag is treated as general.
#[derive(Debug, Deserialize)]
struct DinoVocab {
    idx2tag: Vec<String>,
    #[serde(default)]
    idx2cat: Vec<u8>,
}

// ── Label loaders ─────────────────────────────────────────────────────────────

fn load_wd14_labels(
    repo_name: &str,
    tag_file: &str,
) -> (Vec<String>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let api = Api::new().unwrap();
    let repo = api.model(repo_name.to_string());
    let path = repo.get(tag_file).unwrap();
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap();

    let (mut names, mut ratings, mut general, mut character) = (vec![], vec![], vec![], vec![]);

    for (idx, res) in rdr.deserialize().enumerate() {
        let row: Wd14TagRow = res.unwrap();
        let name = normalise_tag(&row.name);
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

fn load_dino_labels(
    base_repo: &str,
    vocab_file: &str,
) -> (Vec<String>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let api = Api::new().unwrap();
    let repo = api.model(base_repo.to_string());
    let path = repo.get(vocab_file).unwrap();
    let text = std::fs::read_to_string(path).unwrap();
    let vocab: DinoVocab = serde_json::from_str(&text).unwrap();

    let (mut names, mut ratings, mut general, mut character) = (vec![], vec![], vec![], vec![]);

    for (idx, tag) in vocab.idx2tag.iter().enumerate() {
        if !vocab.idx2cat.is_empty() {
            match vocab.idx2cat[idx] {
                9 => ratings.push(idx),
                0 => general.push(idx),
                4 => character.push(idx),
                _ => {}
            }
        } else {
            // No category data: bucket everything as general
            general.push(idx);
        }
        names.push(normalise_tag(tag));
    }
    (names, ratings, general, character)
}

/// Replaces underscores with spaces unless the name is made entirely of
/// punctuation/digits (e.g. "1girl", "^_^") — matching WD14 convention.
fn normalise_tag(name: &str) -> String {
    if name.chars().all(|c| "_()<>+^.0123456789".contains(c)) {
        name.to_string()
    } else {
        name.replace('_', " ")
    }
}

// ── Image preprocessing ───────────────────────────────────────────────────────

/// WD14: centre-pad to square → fixed resize → BGR channel order
/// Output layout: BHWC `[1, H, W, 3]`, raw u8 cast to f32.
fn prepare_wd14(img: &DynamicImage, size: usize) -> Array4<f32> {
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let m = w.max(h);
    let mut canvas = RgbaImage::new(m, m);
    canvas.copy_from(&rgba, (m - w) / 2, (m - h) / 2).unwrap();
    let resized = image::imageops::resize(
        &canvas,
        size as u32,
        size as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let rgb = DynamicImage::ImageRgba8(resized).to_rgb8();

    let mut arr = Array4::<f32>::zeros((1, size, size, 3));
    for y in 0..size {
        for x in 0..size {
            // FIX: original erroneously started at 1, skipping column 0
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                arr[(0, y, x, c)] = pixel[2 - c] as f32; // RGB → BGR
            }
        }
    }
    arr
}

/// DINOv3: aspect-preserving resize (long edge ≤ 1024 px), both dims snapped
/// to multiples of 16, ImageNet-normalised.
/// Output layout: NCHW `[1, 3, H, W]`, float32.
fn prepare_dinov3(img: &DynamicImage) -> Array4<f32> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();

    let scale = (DINO_MAX_SIZE as f32 / w.max(h) as f32).min(1.0);
    // Snap a dimension to the nearest multiple of PATCH_SIZE (min 1 patch).
    let snap = |x: u32| -> u32 {
        DINO_PATCH_SIZE.max(((x as f32 * scale).round() as u32 / DINO_PATCH_SIZE) * DINO_PATCH_SIZE)
    };
    let new_w = snap(w);
    let new_h = snap(h);

    let resized =
        image::imageops::resize(&rgb, new_w, new_h, image::imageops::FilterType::Lanczos3);

    let mut arr = Array4::<f32>::zeros((1, 3, new_h as usize, new_w as usize));
    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                arr[(0, c, y, x)] = (pixel[c] as f32 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    arr
}

// ── Post-processing helpers ───────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn mcut_threshold(probs: &mut [f32]) -> f32 {
    probs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    probs
        .windows(2)
        .enumerate()
        .max_by(|a, b| (a.1[0] - a.1[1]).partial_cmp(&(b.1[0] - b.1[1])).unwrap())
        .map(|(_t, w)| (w[0] + w[1]) / 2.0)
        .unwrap_or(0.0)
}

/// Shared output-building logic for both backends.
fn build_output(
    scores: &[f32],
    embedding: Vec<f32>,
    tag_names: &[String],
    rating_i: &[usize],
    general_i: &[usize],
    character_i: &[usize],
    g_th: f32,
    g_mcut: bool,
    c_th: f32,
    c_mcut: bool,
    exclude: &[String],
) -> OutputData {
    let rating = rating_i
        .iter()
        .map(|&i| (tag_names[i].clone(), scores[i]))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or_else(|| (String::new(), 0.0));

    let mut general: Vec<(String, f32)> = general_i
        .iter()
        .map(|&i| (tag_names[i].clone(), scores[i]))
        .collect();
    let mut character: Vec<(String, f32)> = character_i
        .iter()
        .map(|&i| (tag_names[i].clone(), scores[i]))
        .collect();

    if g_mcut {
        let mut gp: Vec<f32> = general.iter().map(|(_, v)| *v).collect();
        let thresh = mcut_threshold(&mut gp);
        general.retain(|(_, v)| *v > thresh);
    } else {
        general.retain(|(_, v)| *v > g_th);
    }
    general.retain(|(name, _)| !exclude.contains(name));

    if c_mcut {
        let mut cp: Vec<f32> = character.iter().map(|(_, v)| *v).collect();
        let thresh = mcut_threshold(&mut cp).max(0.15);
        character.retain(|(_, v)| *v > thresh);
    } else {
        character.retain(|(_, v)| *v > c_th);
    }
    character.retain(|(name, _)| !exclude.contains(name));

    general.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    OutputData {
        general,
        rating,
        characters: character,
        embedding,
    }
}

// ── Internal loaded-model state ───────────────────────────────────────────────

enum LoadedModel {
    Wd14 {
        session: Session,
        size: usize, // model's expected square input edge, e.g. 448
    },
    DINOv3 {
        tagger: Session,
        embedder: Option<Session>, // None if embedding/model.onnx is unavailable
    },
}

// ── Public Predictor ──────────────────────────────────────────────────────────

pub struct Predictor {
    kind: ModelKind,
    model: Option<LoadedModel>,
    tag_names: Vec<String>,
    rating_i: Vec<usize>,
    general_i: Vec<usize>,
    character_i: Vec<usize>,
}

impl Predictor {
    pub fn new(kind: ModelKind) -> Self {
        Predictor {
            kind,
            model: None,
            tag_names: vec![],
            rating_i: vec![],
            general_i: vec![],
            character_i: vec![],
        }
    }

    fn ensure_loaded(&mut self) {
        if self.model.is_some() {
            return;
        }
        match self.kind {
            ModelKind::Wd14 => self.load_wd14(),
            ModelKind::DINOv3 => self.load_dinov3(),
        }
    }

    fn load_wd14(&mut self) {
        let api = Api::new().unwrap();
        let repo = api.model(WD14_REPO.to_string());
        let model_path = repo.get(WD14_MODEL_FILE).unwrap();

        let (tags, r, g, c) = load_wd14_labels(WD14_REPO, WD14_TAG_CSV);
        self.tag_names = tags;
        self.rating_i = r;
        self.general_i = g;
        self.character_i = c;

        let session = Session::builder()
            .unwrap()
            .with_execution_providers([ep::CUDA::default().build()])
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        let size = match session.inputs()[0].dtype() {
            ort::value::ValueType::Tensor { shape, .. } => shape[2] as usize,
            _ => panic!("WD14: expected a tensor input"),
        };

        self.model = Some(LoadedModel::Wd14 { session, size });
    }

    fn load_dinov3(&mut self) {
        let api = Api::new().unwrap();
        let onnx_repo = api.model(DINO_ONNX_REPO.to_string());

        // hf_hub preserves the repo's directory tree under
        // ~/.cache/huggingface/hub/.../snapshots/<rev>/
        // so model.onnx and model.onnx.data land in the same folder,
        // which is what ONNX Runtime needs to find the external-data sidecar.
        let tagger_path = onnx_repo.get(DINO_TAGGER_MODEL).unwrap();
        let _ = onnx_repo.get(DINO_TAGGER_DATA).unwrap();

        let (tags, r, g, c) = load_dino_labels(DINO_BASE_REPO, DINO_VOCAB_FILE);
        self.tag_names = tags;
        self.rating_i = r;
        self.general_i = g;
        self.character_i = c;

        let tagger = Session::builder()
            .unwrap()
            .with_execution_providers([ep::CUDA::default().build()])
            .unwrap()
            .commit_from_file(tagger_path)
            .unwrap();

        // Embedder is optional; skip cleanly if the files aren't present.
        let embedder = onnx_repo.get(DINO_EMBED_MODEL).ok().map(|emb_path| {
            let _ = onnx_repo.get(DINO_EMBED_DATA).ok();
            Session::builder()
                .unwrap()
                .with_execution_providers([ep::CUDA::default().build()])
                .unwrap()
                .commit_from_file(emb_path)
                .unwrap()
        });

        self.model = Some(LoadedModel::DINOv3 { tagger, embedder });
    }

    pub fn predict(
        &mut self,
        img: &DynamicImage,
        g_th: f32,
        g_mcut: bool,
        c_th: f32,
        c_mcut: bool,
        exclude: &[String],
    ) -> OutputData {
        self.ensure_loaded();

        // Produce (probabilities, embedding) — both owned Vecs — before
        // touching any other field of self.
        let (scores, embedding): (Vec<f32>, Vec<f32>) = match self.model.as_mut().unwrap() {
            LoadedModel::Wd14 { session, size } => {
                let arr = prepare_wd14(img, *size);
                let outputs = session
                    .run(ort::inputs![Tensor::from_array(arr).unwrap()])
                    .unwrap();

                let pred_raw = outputs[0].try_extract_array::<f32>().unwrap();
                let scores: Vec<f32> = pred_raw
                    .view()
                    .into_dimensionality::<Ix2>()
                    .unwrap()
                    .row(0)
                    .to_vec();

                let emb_raw: Vec<f32> = outputs[1]
                    .try_extract_array::<f32>()
                    .unwrap()
                    .iter()
                    .copied()
                    .collect();

                (scores, emb_raw)
            }

            LoadedModel::DINOv3 { tagger, embedder } => {
                let arr = prepare_dinov3(img);

                // Tag predictions — model outputs raw logits, apply sigmoid here.
                let tag_out = tagger
                    .run(ort::inputs![Tensor::from_array(arr.clone()).unwrap()])
                    .unwrap();
                let logit_raw = tag_out[0].try_extract_array::<f32>().unwrap();
                let scores: Vec<f32> = logit_raw
                    .view()
                    .into_dimensionality::<Ix2>()
                    .unwrap()
                    .row(0)
                    .iter()
                    .map(|&x| sigmoid(x))
                    .collect();

                // Embeddings from the separate embedder session (if loaded).
                let embedding: Vec<f32> = if let Some(emb_sess) = embedder {
                    let emb_out = emb_sess
                        .run(ort::inputs![Tensor::from_array(arr).unwrap()])
                        .unwrap();
                    emb_out[0]
                        .try_extract_array::<f32>()
                        .unwrap()
                        .iter()
                        .copied()
                        .collect()
                } else {
                    vec![]
                };

                (scores, embedding)
            }
        };
        // ↑ Both borrows of self.model end here; subsequent borrows are safe.

        build_output(
            &scores,
            embedding,
            &self.tag_names,
            &self.rating_i,
            &self.general_i,
            &self.character_i,
            g_th,
            g_mcut,
            c_th,
            c_mcut,
            exclude,
        )
    }
}
