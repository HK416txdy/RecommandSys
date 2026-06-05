use std::collections::HashMap;

use crate::data::DataModel;
use crate::math::{clamp01, dot, rating_to_norm};
use crate::recommender::{Recommender, recommend_from};
use crate::types::{ItemId, Recommendation, ScoredRecommendation, UserId};

#[derive(Clone, Debug)]
pub struct MatrixFactorization {
    user_index: HashMap<UserId, usize>,
    item_index: HashMap<ItemId, usize>,
    user_bias: Vec<f32>,
    item_bias: Vec<f32>,
    user_factors: Vec<Vec<f32>>,
    item_factors: Vec<Vec<f32>>,
    global_mean: f32,
}

impl MatrixFactorization {
    pub fn train(model: &DataModel, factors: usize, epochs: usize, lr: f32, reg: f32) -> Self {
        let mut users: Vec<UserId> = model.by_user.keys().copied().collect();
        let mut items: Vec<ItemId> = model.by_item.keys().copied().collect();
        users.sort_unstable();
        items.sort_unstable();

        let user_index = users
            .iter()
            .enumerate()
            .map(|(idx, id)| (*id, idx))
            .collect::<HashMap<_, _>>();
        let item_index = items
            .iter()
            .enumerate()
            .map(|(idx, id)| (*id, idx))
            .collect::<HashMap<_, _>>();
        let mut user_bias = vec![0.0; users.len()];
        let mut item_bias = vec![0.0; items.len()];
        let mut user_factors = vec![vec![0.0; factors]; users.len()];
        let mut item_factors = vec![vec![0.0; factors]; items.len()];

        for (user, row) in user_factors.iter_mut().enumerate() {
            for (factor, value) in row.iter_mut().enumerate() {
                *value = seeded_value(user, factor);
            }
        }
        for (item, row) in item_factors.iter_mut().enumerate() {
            for (factor, value) in row.iter_mut().enumerate() {
                *value = seeded_value(item + 17, factor);
            }
        }

        for _ in 0..epochs {
            for rating in &model.dataset.ratings {
                let Some(&u) = user_index.get(&rating.user_id) else {
                    continue;
                };
                let Some(&i) = item_index.get(&rating.movie_id) else {
                    continue;
                };
                let pred = model.global_mean
                    + user_bias[u]
                    + item_bias[i]
                    + dot(&user_factors[u], &item_factors[i]);
                let err = rating.rating - pred;
                user_bias[u] += lr * (err - reg * user_bias[u]);
                item_bias[i] += lr * (err - reg * item_bias[i]);
                for factor in 0..factors {
                    let uf = user_factors[u][factor];
                    let mf = item_factors[i][factor];
                    user_factors[u][factor] += lr * (err * mf - reg * uf);
                    item_factors[i][factor] += lr * (err * uf - reg * mf);
                }
            }
        }

        Self {
            user_index,
            item_index,
            user_bias,
            item_bias,
            user_factors,
            item_factors,
            global_mean: model.global_mean,
        }
    }

    pub fn predict(&self, user_id: UserId, item_id: ItemId) -> Option<f32> {
        let &user = self.user_index.get(&user_id)?;
        let &item = self.item_index.get(&item_id)?;
        Some(
            (self.global_mean
                + self.user_bias[user]
                + self.item_bias[item]
                + dot(&self.user_factors[user], &self.item_factors[item]))
            .clamp(1.0, 5.0),
        )
    }
}

pub struct MatrixRecommender<'a> {
    model: &'a DataModel,
    mf: MatrixFactorization,
}

impl<'a> MatrixRecommender<'a> {
    pub fn new(model: &'a DataModel) -> Self {
        Self {
            model,
            mf: MatrixFactorization::train(model, 24, 20, 0.01, 0.05),
        }
    }
}

impl Recommender for MatrixRecommender<'_> {
    fn score(&self, user_id: UserId, item_id: ItemId) -> Option<ScoredRecommendation> {
        let prediction = self.mf.predict(user_id, item_id)?;
        let user_count = self
            .model
            .by_user
            .get(&user_id)
            .map(|r| r.len())
            .unwrap_or(0);
        let item_count = self
            .model
            .by_item
            .get(&item_id)
            .map(|r| r.len())
            .unwrap_or(0);
        Some(ScoredRecommendation {
            raw_score: prediction,
            normalized_score: rating_to_norm(prediction),
            confidence: clamp01(
                (user_count as f32 / 10.0).min(1.0) * (item_count as f32 / 10.0).min(1.0),
            ),
            reason: "matrix factorization latent-vector prediction".to_string(),
        })
    }

    fn recommend(&self, user_id: UserId, top_n: usize) -> Vec<Recommendation> {
        recommend_from(self, self.model, user_id, top_n)
    }
}

fn seeded_value(a: usize, b: usize) -> f32 {
    let n = ((a as u64 + 1) * 1_103_515_245 + (b as u64 + 7) * 12_345) % 1000;
    (n as f32 / 1000.0 - 0.5) * 0.08
}
