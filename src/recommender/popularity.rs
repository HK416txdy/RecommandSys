use std::collections::HashMap;

use crate::data::DataModel;
use crate::math::{clamp01, rating_to_norm};
use crate::recommender::{Recommender, recommend_from};
use crate::types::{ItemId, Recommendation, ScoredRecommendation, UserId};

pub struct PopularityRecommender<'a> {
    model: &'a DataModel,
    scores: HashMap<ItemId, ScoredRecommendation>,
}

impl<'a> PopularityRecommender<'a> {
    pub fn new(model: &'a DataModel) -> Self {
        let scores = model
            .by_item
            .iter()
            .map(|(item_id, ratings)| {
                let count = ratings.len() as f32;
                let average = ratings.iter().map(|r| r.rating).sum::<f32>() / count.max(1.0);
                let prior_count = 5.0;
                let smoothed =
                    (average * count + model.global_mean * prior_count) / (count + prior_count);
                (
                    *item_id,
                    ScoredRecommendation {
                        raw_score: smoothed,
                        normalized_score: rating_to_norm(smoothed),
                        confidence: clamp01(count / 20.0),
                        reason: format!(
                            "Bayesian-smoothed popularity from {} ratings",
                            count as usize
                        ),
                    },
                )
            })
            .collect();
        Self { model, scores }
    }
}

impl Recommender for PopularityRecommender<'_> {
    fn score(&self, _user_id: UserId, item_id: ItemId) -> Option<ScoredRecommendation> {
        self.scores.get(&item_id).cloned()
    }

    fn recommend(&self, user_id: UserId, top_n: usize) -> Vec<Recommendation> {
        recommend_from(self, self.model, user_id, top_n)
    }
}
