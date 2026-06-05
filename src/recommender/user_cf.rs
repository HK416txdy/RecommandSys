use std::collections::HashMap;

use crate::data::DataModel;
use crate::math::{clamp01, pearson, rating_to_norm};
use crate::recommender::{Recommender, recommend_from};
use crate::types::{ItemId, Recommendation, ScoredRecommendation, UserId};

pub struct UserCfRecommender<'a> {
    model: &'a DataModel,
    similarities: HashMap<(UserId, UserId), f32>,
}

impl<'a> UserCfRecommender<'a> {
    pub fn new(model: &'a DataModel) -> Self {
        let users = model.by_user.keys().copied().collect::<Vec<_>>();
        let rating_maps = model
            .by_user
            .iter()
            .map(|(user_id, ratings)| {
                (
                    *user_id,
                    ratings
                        .iter()
                        .map(|rating| (rating.movie_id, rating.rating))
                        .collect::<HashMap<_, _>>(),
                )
            })
            .collect::<HashMap<_, _>>();
        let mut similarities = HashMap::new();
        for i in 0..users.len() {
            for j in (i + 1)..users.len() {
                let sim = user_similarity(users[i], users[j], &rating_maps);
                if sim > 0.0 {
                    similarities.insert((users[i], users[j]), sim);
                    similarities.insert((users[j], users[i]), sim);
                }
            }
        }
        Self {
            model,
            similarities,
        }
    }

    fn similarity(&self, a: UserId, b: UserId) -> f32 {
        self.similarities.get(&(a, b)).copied().unwrap_or(0.0)
    }
}

fn user_similarity(
    a: UserId,
    b: UserId,
    rating_maps: &HashMap<UserId, HashMap<ItemId, f32>>,
) -> f32 {
    let Some(a_ratings) = rating_maps.get(&a) else {
        return 0.0;
    };
    let Some(b_ratings) = rating_maps.get(&b) else {
        return 0.0;
    };
    let mut av = Vec::new();
    let mut bv = Vec::new();
    for (movie_id, rating) in a_ratings {
        if let Some(other) = b_ratings.get(movie_id) {
            av.push(*rating);
            bv.push(*other);
        }
    }
    if av.len() < 2 {
        0.0
    } else {
        pearson(&av, &bv).max(0.0)
    }
}

impl Recommender for UserCfRecommender<'_> {
    fn score(&self, user_id: UserId, item_id: ItemId) -> Option<ScoredRecommendation> {
        let item_ratings = self.model.by_item.get(&item_id)?;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let mut neighbors = 0usize;

        for rating in item_ratings {
            if rating.user_id == user_id {
                continue;
            }
            let sim = self.similarity(user_id, rating.user_id);
            if sim > 0.0 {
                numerator += sim * rating.rating;
                denominator += sim.abs();
                neighbors += 1;
            }
        }

        let prediction = if denominator > 0.0 {
            numerator / denominator
        } else {
            self.model.global_mean
        };
        Some(ScoredRecommendation {
            raw_score: prediction,
            normalized_score: rating_to_norm(prediction),
            confidence: clamp01((neighbors as f32 / 8.0) * denominator.min(1.0)),
            reason: format!("predicted from {neighbors} similar users"),
        })
    }

    fn recommend(&self, user_id: UserId, top_n: usize) -> Vec<Recommendation> {
        recommend_from(self, self.model, user_id, top_n)
    }
}
