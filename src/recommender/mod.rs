use std::cmp::Ordering;

use crate::data::DataModel;
use crate::math::norm_to_rating;
use crate::types::{ItemId, Recommendation, ScoredRecommendation, UserId};

mod content;
mod item_cf;
mod knowledge;
mod matrix;
mod popularity;
mod user_cf;

pub use content::{ContentRecommender, movie_features};
pub use item_cf::ItemCfRecommender;
pub use knowledge::KnowledgeRecommender;
pub use matrix::{MatrixFactorization, MatrixRecommender};
pub use popularity::PopularityRecommender;
pub use user_cf::UserCfRecommender;

pub trait Recommender {
    fn score(&self, user_id: UserId, item_id: ItemId) -> Option<ScoredRecommendation>;

    fn recommend(&self, user_id: UserId, top_n: usize) -> Vec<Recommendation>;
}

pub fn build_recommender<'a>(name: &str, model: &'a DataModel) -> Box<dyn Recommender + 'a> {
    match name {
        "content" => Box::new(ContentRecommender::new(model)),
        "knowledge" => Box::new(KnowledgeRecommender::new(model)),
        "user-cf" => Box::new(UserCfRecommender::new(model)),
        "item-cf" => Box::new(ItemCfRecommender::new(model)),
        "popularity" => Box::new(PopularityRecommender::new(model)),
        "matrix" => Box::new(MatrixRecommender::new(model)),
        "hybrid" => Box::new(crate::blending::HybridRecommender::new(model)),
        _ => Box::new(crate::blending::HybridRecommender::new(model)),
    }
}

pub(crate) fn recommend_from<R: Recommender>(
    recommender: &R,
    model: &DataModel,
    user_id: UserId,
    top_n: usize,
) -> Vec<Recommendation> {
    let seen = model.user_seen(user_id);
    let mut recs = Vec::new();
    for item_id in model.dataset.movies.keys().copied() {
        if seen.contains(&item_id) {
            continue;
        }
        if let Some(score) = recommender.score(user_id, item_id) {
            recs.push(Recommendation {
                movie_id: item_id,
                title: model.movie_title(item_id),
                score: score.normalized_score,
                predicted_rating: norm_to_rating(score.normalized_score),
                reason: score.reason,
                weights: None,
            });
        }
    }
    sort_recommendations(&mut recs);
    recs.truncate(top_n);
    recs
}

pub(crate) fn sort_recommendations(recs: &mut [Recommendation]) {
    recs.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.movie_id.cmp(&b.movie_id))
    });
}
