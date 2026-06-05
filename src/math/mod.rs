use std::collections::HashMap;

pub fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

pub fn rating_to_norm(rating: f32) -> f32 {
    clamp01((rating - 1.0) / 4.0)
}

pub fn norm_to_rating(norm: f32) -> f32 {
    1.0 + 4.0 * clamp01(norm)
}

pub fn cosine_sparse(a: &HashMap<String, f32>, b: &HashMap<String, f32>) -> f32 {
    let dot = a
        .iter()
        .map(|(key, av)| av * b.get(key).unwrap_or(&0.0))
        .sum::<f32>();
    let an = a.values().map(|v| v * v).sum::<f32>().sqrt();
    let bn = b.values().map(|v| v * v).sum::<f32>().sqrt();
    if an <= f32::EPSILON || bn <= f32::EPSILON {
        0.0
    } else {
        dot / (an * bn)
    }
}

pub fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let am = a.iter().take(n).sum::<f32>() / n as f32;
    let bm = b.iter().take(n).sum::<f32>() / n as f32;
    let mut num = 0.0;
    let mut ad = 0.0;
    let mut bd = 0.0;
    for i in 0..n {
        let da = a[i] - am;
        let db = b[i] - bm;
        num += da * db;
        ad += da * da;
        bd += db * db;
    }
    if ad <= f32::EPSILON || bd <= f32::EPSILON {
        0.0
    } else {
        num / (ad.sqrt() * bd.sqrt())
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
