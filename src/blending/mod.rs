use crate::data::DataModel;
use crate::math::norm_to_rating;
use crate::recommender::{
    ContentRecommender, ItemCfRecommender, KnowledgeRecommender, MatrixRecommender,
    PopularityRecommender, Recommender, UserCfRecommender, sort_recommendations,
};
use crate::types::{AlgorithmWeights, ItemId, Recommendation, ScoredRecommendation, UserId};

pub struct HybridRecommender<'a> {
    model: &'a DataModel,
    content: ContentRecommender<'a>,
    knowledge: KnowledgeRecommender<'a>,
    user_cf: UserCfRecommender<'a>,
    item_cf: ItemCfRecommender<'a>,
    popularity: PopularityRecommender<'a>,
    matrix: MatrixRecommender<'a>,
}

impl<'a> HybridRecommender<'a> {
    pub fn new(model: &'a DataModel) -> Self {
        Self {
            model,
            content: ContentRecommender::new(model),
            knowledge: KnowledgeRecommender::new(model),
            user_cf: UserCfRecommender::new(model),
            item_cf: ItemCfRecommender::new(model),
            popularity: PopularityRecommender::new(model),
            matrix: MatrixRecommender::new(model),
        }
    }

    fn dynamic_weights(
        &self,
        user_id: UserId,
        item_id: ItemId,
        scores: [&Option<ScoredRecommendation>; 6],
    ) -> AlgorithmWeights {
        let user_count = self
            .model
            .by_user
            .get(&user_id)
            .map(|r| r.len())
            .unwrap_or(0) as f32;
        let item_count = self
            .model
            .by_item
            .get(&item_id)
            .map(|r| r.len())
            .unwrap_or(0) as f32;
        let cold_user_boost = if user_count < 3.0 { 1.5 } else { 1.0 };
        let sparse_item_penalty = if item_count < 3.0 { 0.65 } else { 1.0 };

        AlgorithmWeights {
            content: 0.22 * confidence(scores[0]) * cold_user_boost,
            knowledge: 0.18 * confidence_floor(scores[1], 0.2) * cold_user_boost,
            user_cf: 0.18 * confidence(scores[2]) * sparse_item_penalty,
            item_cf: 0.18 * confidence(scores[3]) * sparse_item_penalty,
            popularity: 0.10
                * confidence_floor(scores[4], 0.2)
                * if user_count < 3.0 { 1.4 } else { 1.0 },
            matrix: 0.14 * confidence(scores[5]) * sparse_item_penalty,
        }
        .normalize()
    }

    fn component_scores(
        &self,
        user_id: UserId,
        item_id: ItemId,
    ) -> [Option<ScoredRecommendation>; 6] {
        [
            self.content.score(user_id, item_id),
            self.knowledge.score(user_id, item_id),
            self.user_cf.score(user_id, item_id),
            self.item_cf.score(user_id, item_id),
            self.popularity.score(user_id, item_id),
            self.matrix.score(user_id, item_id),
        ]
    }

    fn blended_score(
        &self,
        scores: &[Option<ScoredRecommendation>; 6],
        weights: AlgorithmWeights,
    ) -> f32 {
        scores[0]
            .as_ref()
            .map(|s| s.normalized_score)
            .unwrap_or(0.0)
            * weights.content
            + scores[1]
                .as_ref()
                .map(|s| s.normalized_score)
                .unwrap_or(0.0)
                * weights.knowledge
            + scores[2]
                .as_ref()
                .map(|s| s.normalized_score)
                .unwrap_or(0.0)
                * weights.user_cf
            + scores[3]
                .as_ref()
                .map(|s| s.normalized_score)
                .unwrap_or(0.0)
                * weights.item_cf
            + scores[4]
                .as_ref()
                .map(|s| s.normalized_score)
                .unwrap_or(0.0)
                * weights.popularity
            + scores[5]
                .as_ref()
                .map(|s| s.normalized_score)
                .unwrap_or(0.0)
                * weights.matrix
    }

    fn reason(weights: AlgorithmWeights) -> String {
        format!(
            "dynamic hybrid, dominant {}; weights c={:.2}, k={:.2}, ucf={:.2}, icf={:.2}, pop={:.2}, mf={:.2}",
            weights.dominant(),
            weights.content,
            weights.knowledge,
            weights.user_cf,
            weights.item_cf,
            weights.popularity,
            weights.matrix
        )
    }
}

impl Recommender for HybridRecommender<'_> {
    fn score(&self, user_id: UserId, item_id: ItemId) -> Option<ScoredRecommendation> {
        let scores = self.component_scores(user_id, item_id);
        let weights = self.dynamic_weights(
            user_id,
            item_id,
            [
                &scores[0], &scores[1], &scores[2], &scores[3], &scores[4], &scores[5],
            ],
        );
        let blended = self.blended_score(&scores, weights);

        Some(ScoredRecommendation {
            raw_score: norm_to_rating(blended),
            normalized_score: blended,
            confidence: weights.sum(),
            reason: Self::reason(weights),
        })
    }

    fn recommend(&self, user_id: UserId, top_n: usize) -> Vec<Recommendation> {
        let seen = self.model.user_seen(user_id);
        let mut recs = Vec::new();
        for item_id in self.model.dataset.movies.keys().copied() {
            if seen.contains(&item_id) {
                continue;
            }
            let scores = self.component_scores(user_id, item_id);
            let weights = self.dynamic_weights(
                user_id,
                item_id,
                [
                    &scores[0], &scores[1], &scores[2], &scores[3], &scores[4], &scores[5],
                ],
            );
            let score = self.blended_score(&scores, weights);
            recs.push(Recommendation {
                movie_id: item_id,
                title: self.model.movie_title(item_id),
                score,
                predicted_rating: norm_to_rating(score),
                reason: Self::reason(weights),
                weights: Some(weights),
            });
        }
        sort_recommendations(&mut recs);
        recs.truncate(top_n);
        recs
    }
}

fn confidence(score: &Option<ScoredRecommendation>) -> f32 {
    score.as_ref().map(|s| s.confidence).unwrap_or(0.0)
}

fn confidence_floor(score: &Option<ScoredRecommendation>, floor: f32) -> f32 {
    score
        .as_ref()
        .map(|s| s.confidence.max(floor))
        .unwrap_or(0.0)
}
