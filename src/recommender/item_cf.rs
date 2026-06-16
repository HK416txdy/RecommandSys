use std::collections::HashMap;

use crate::data::DataModel;
use crate::math::{clamp01, pearson, rating_to_norm};
use crate::recommender::{Recommender, recommend_from};
use crate::types::{ItemId, Recommendation, ScoredRecommendation, UserId};

#[derive(Clone, Copy, Debug)]
struct Similarity {
    value: f32,
    overlap: usize,
}

pub struct ItemCfRecommender<'a> {
    model: &'a DataModel,
    similarities: HashMap<(ItemId, ItemId), Similarity>,
}

impl<'a> ItemCfRecommender<'a> {
    pub fn new(model: &'a DataModel) -> Self {
        let items = model.by_item.keys().copied().collect::<Vec<_>>();
        let rating_maps = model
            .by_item
            .iter()
            .map(|(item_id, ratings)| {
                (
                    *item_id,
                    ratings
                        .iter()
                        .map(|rating| (rating.user_id, rating.rating))
                        .collect::<HashMap<_, _>>(),
                )
            })
            .collect::<HashMap<_, _>>();
        let mut similarities = HashMap::new();
        for i in 0..items.len() {
            for j in (i + 1)..items.len() {
                let sim = item_similarity(items[i], items[j], &rating_maps);
                if sim.value > 0.0 {
                    similarities.insert((items[i], items[j]), sim);
                    similarities.insert((items[j], items[i]), sim);
                }
            }
        }
        Self {
            model,
            similarities,
        }
    }

    fn similarity(&self, a: ItemId, b: ItemId) -> Option<Similarity> {
        self.similarities.get(&(a, b)).copied()
    }
}

fn item_similarity(
    a: ItemId,
    b: ItemId,
    rating_maps: &HashMap<ItemId, HashMap<UserId, f32>>,
) -> Similarity {
    let Some(a_ratings) = rating_maps.get(&a) else {
        return Similarity {
            value: 0.0,
            overlap: 0,
        };
    };
    let Some(b_ratings) = rating_maps.get(&b) else {
        return Similarity {
            value: 0.0,
            overlap: 0,
        };
    };
    let mut av = Vec::new();
    let mut bv = Vec::new();
    for (user_id, rating) in a_ratings {
        if let Some(other) = b_ratings.get(user_id) {
            av.push(*rating);
            bv.push(*other);
        }
    }
    if av.len() < 2 {
        Similarity {
            value: 0.0,
            overlap: av.len(),
        }
    } else {
        let overlap = av.len();
        let shrink = overlap as f32 / (overlap as f32 + 15.0);
        Similarity {
            value: pearson(&av, &bv).max(0.0) * shrink,
            overlap,
        }
    }
}

impl Recommender for ItemCfRecommender<'_> {
    fn score(&self, user_id: UserId, item_id: ItemId) -> Option<ScoredRecommendation> {
        let user_ratings = self.model.by_user.get(&user_id)?;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let mut neighbors = 0usize;

        for rating in user_ratings {
            let Some(sim) = self.similarity(item_id, rating.movie_id) else {
                continue;
            };
            if sim.value > 0.0 {
                let rated_item_mean = self
                    .model
                    .item_mean
                    .get(&rating.movie_id)
                    .copied()
                    .unwrap_or(self.model.global_mean);
                numerator += sim.value * (rating.rating - rated_item_mean);
                denominator += sim.value.abs();
                if sim.overlap >= 5 {
                    neighbors += 1;
                }
            }
        }

        let item_mean = self
            .model
            .item_mean
            .get(&item_id)
            .copied()
            .unwrap_or(self.model.global_mean);
        let prediction = if denominator > 0.0 {
            let adjustment = numerator / denominator;
            let confidence = clamp01((neighbors as f32 / 20.0) * denominator.min(1.0));
            item_mean + adjustment * confidence
        } else {
            self.model
                .user_mean
                .get(&user_id)
                .copied()
                .unwrap_or(self.model.global_mean)
        }
        .clamp(1.0, 5.0);
        Some(ScoredRecommendation {
            raw_score: prediction,
            normalized_score: rating_to_norm(prediction),
            confidence: clamp01((neighbors as f32 / 20.0) * denominator.min(1.0)),
            reason: format!("predicted from {neighbors} similar items"),
        })
    }

    fn recommend(&self, user_id: UserId, top_n: usize) -> Vec<Recommendation> {
        recommend_from(self, self.model, user_id, top_n)
    }
}
