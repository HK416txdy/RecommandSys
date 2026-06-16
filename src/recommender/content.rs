use std::collections::HashMap;

use crate::data::DataModel;
use crate::math::{clamp01, cosine_sparse, rating_to_norm};
use crate::recommender::{Recommender, recommend_from};
use crate::types::{ItemId, Movie, Recommendation, ScoredRecommendation, UserId};

pub struct ContentRecommender<'a> {
    model: &'a DataModel,
    movie_features: HashMap<ItemId, HashMap<String, f32>>,
    user_profiles: HashMap<UserId, HashMap<String, f32>>,
}

impl<'a> ContentRecommender<'a> {
    pub fn new(model: &'a DataModel) -> Self {
        let movie_features = model
            .dataset
            .movies
            .iter()
            .map(|(item_id, movie)| (*item_id, movie_features(movie)))
            .collect::<HashMap<_, _>>();
        let mut user_profiles = HashMap::new();
        for (user_id, ratings) in &model.by_user {
            let mut profile = HashMap::new();
            for rating in ratings.iter().filter(|r| r.rating >= 4.0) {
                if let Some(features) = movie_features.get(&rating.movie_id) {
                    for (key, value) in features {
                        *profile.entry(key.clone()).or_insert(0.0) +=
                            value * rating_to_norm(rating.rating);
                    }
                }
            }
            user_profiles.insert(*user_id, profile);
        }
        Self {
            model,
            movie_features,
            user_profiles,
        }
    }
}

impl Recommender for ContentRecommender<'_> {
    fn score(&self, user_id: UserId, item_id: ItemId) -> Option<ScoredRecommendation> {
        let features = self.movie_features.get(&item_id)?;
        let profile = self.user_profiles.get(&user_id);
        let score = profile
            .map(|profile| cosine_sparse(profile, features))
            .unwrap_or(0.0);
        let history = self
            .model
            .by_user
            .get(&user_id)
            .map(|r| r.len())
            .unwrap_or(0);
        let confidence = clamp01((features.len() as f32 / 8.0) * (history as f32 / 8.0).min(1.0));
        let normalized_score = 0.45 + 0.45 * clamp01(score);
        Some(ScoredRecommendation {
            raw_score: score,
            normalized_score,
            confidence,
            reason: "content features match the user's high-rated profile".to_string(),
        })
    }

    fn recommend(&self, user_id: UserId, top_n: usize) -> Vec<Recommendation> {
        recommend_from(self, self.model, user_id, top_n)
    }
}

pub fn movie_features(movie: &Movie) -> HashMap<String, f32> {
    let mut features = HashMap::new();
    for genre in &movie.genres {
        *features
            .entry(format!("genre:{}", genre.to_lowercase()))
            .or_insert(0.0) += 2.0;
    }
    for token in movie
        .title
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() > 2 && !t.chars().all(|c| c.is_ascii_digit()))
    {
        *features.entry(format!("title:{token}")).or_insert(0.0) += 0.5;
    }
    if let Some(year) = movie.release_year {
        *features
            .entry(format!("decade:{}", year / 10 * 10))
            .or_insert(0.0) += 0.6;
    }
    features
}
