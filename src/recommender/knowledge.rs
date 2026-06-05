use std::collections::{HashMap, HashSet};

use crate::data::DataModel;
use crate::math::clamp01;
use crate::recommender::{Recommender, recommend_from};
use crate::types::{ItemId, Recommendation, ScoredRecommendation, UserId};

#[derive(Clone, Debug, Default)]
struct UserPreferences {
    liked: HashMap<String, f32>,
    rejected: HashSet<String>,
    year_range: Option<(u16, u16)>,
}

pub struct KnowledgeRecommender<'a> {
    model: &'a DataModel,
    preferences: HashMap<UserId, UserPreferences>,
}

impl<'a> KnowledgeRecommender<'a> {
    pub fn new(model: &'a DataModel) -> Self {
        let preferences = model
            .by_user
            .keys()
            .copied()
            .map(|user_id| (user_id, build_preferences(model, user_id)))
            .collect();
        Self { model, preferences }
    }
}

fn build_preferences(model: &DataModel, user_id: UserId) -> UserPreferences {
    let mut liked = HashMap::new();
    let mut disliked = HashMap::new();
    let mut liked_years = Vec::new();
    if let Some(ratings) = model.by_user.get(&user_id) {
        for rating in ratings {
            if let Some(movie) = model.dataset.movies.get(&rating.movie_id) {
                if rating.rating >= 4.0 {
                    for genre in &movie.genres {
                        *liked.entry(genre.clone()).or_insert(0.0) += rating.rating;
                    }
                    if let Some(year) = movie.release_year {
                        liked_years.push(year);
                    }
                } else if rating.rating <= 2.0 {
                    for genre in &movie.genres {
                        *disliked.entry(genre.clone()).or_insert(0usize) += 1;
                    }
                }
            }
        }
    }
    let rejected = disliked
        .into_iter()
        .filter(|(_, count)| *count >= 1)
        .map(|(genre, _)| genre)
        .collect();
    liked_years.sort_unstable();
    let year_range = liked_years
        .first()
        .zip(liked_years.last())
        .map(|(a, b)| (*a, *b));
    UserPreferences {
        liked,
        rejected,
        year_range,
    }
}

impl Recommender for KnowledgeRecommender<'_> {
    fn score(&self, user_id: UserId, item_id: ItemId) -> Option<ScoredRecommendation> {
        let movie = self.model.dataset.movies.get(&item_id)?;
        let preferences = self.preferences.get(&user_id);
        let mut rule_hits = 0.0;
        let mut score = 0.35;
        let mut reasons = Vec::new();

        for genre in &movie.genres {
            if let Some(weight) = preferences.and_then(|prefs| prefs.liked.get(genre)) {
                score += 0.12 * *weight;
                rule_hits += 1.0;
                reasons.push(format!("preferred genre {genre}"));
            }
            if preferences
                .map(|prefs| prefs.rejected.contains(genre))
                .unwrap_or(false)
            {
                score -= 0.18;
                reasons.push(format!("penalized disliked genre {genre}"));
            }
        }

        if let (Some((min_year, max_year)), Some(year)) = (
            preferences.and_then(|prefs| prefs.year_range),
            movie.release_year,
        ) && year >= min_year.saturating_sub(5)
            && year <= max_year + 5
        {
            score += 0.12;
            rule_hits += 1.0;
            reasons.push("release year is close to the user's preferred range".to_string());
        }

        Some(ScoredRecommendation {
            raw_score: score,
            normalized_score: clamp01(score / 2.0),
            confidence: clamp01(rule_hits / 4.0),
            reason: if reasons.is_empty() {
                "knowledge rules used default preference".to_string()
            } else {
                reasons.join("; ")
            },
        })
    }

    fn recommend(&self, user_id: UserId, top_n: usize) -> Vec<Recommendation> {
        recommend_from(self, self.model, user_id, top_n)
    }
}
