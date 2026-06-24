use std::collections::{HashMap, HashSet};

use crate::data::DataModel;
use crate::math::norm_to_rating;
use crate::recommender::build_recommender;
use crate::types::{Dataset, ItemId, Rating, Recommendation, UserId};

#[derive(Clone, Debug, Default)]
pub struct EvaluationReport {
    pub prediction_coverage: f32,
    pub prediction_count: usize,
    pub ranking_user_count: usize,
    pub accuracy: f32,
    pub ndcg_at_n: f32,
    pub mae: f32,
    pub rmse: f32,
    pub precision_at_n: f32,
    pub recall_at_n: f32,
    pub f1_at_n: f32,
    pub hit_rate_at_n: f32,
    pub catalog_coverage: f32,
    pub avg_recommendation_popularity: f32,
    pub avg_genre_diversity: f32,
}

pub fn split_holdout(dataset: &Dataset, ratio: f32) -> (Dataset, Vec<Rating>) {
    let mut train = Vec::new();
    let mut test = Vec::new();
    let mut by_user: HashMap<UserId, Vec<Rating>> = HashMap::new();
    for rating in &dataset.ratings {
        by_user
            .entry(rating.user_id)
            .or_default()
            .push(rating.clone());
    }

    for (_user, mut ratings) in by_user {
        ratings.sort_by_key(|r| r.timestamp);
        if ratings.len() == 1 {
            train.push(ratings.remove(0));
            continue;
        }
        let test_n = ((ratings.len() as f32 * ratio).round() as usize).clamp(1, ratings.len() - 1);
        let split = ratings.len() - test_n;
        for (idx, rating) in ratings.into_iter().enumerate() {
            if idx >= split {
                test.push(rating);
            } else {
                train.push(rating);
            }
        }
    }

    (
        Dataset {
            movies: dataset.movies.clone(),
            ratings: train,
        },
        test,
    )
}

pub fn evaluate_algorithm(
    name: &str,
    dataset: &Dataset,
    top_n: usize,
    holdout_ratio: f32,
) -> EvaluationReport {
    let (train, test) = split_holdout(dataset, holdout_ratio);
    let model = DataModel::new(train);
    let users: HashSet<UserId> = test.iter().map(|r| r.user_id).collect();
    let recommender = build_recommender(name, &model);
    let mut abs_err = 0.0;
    let mut sq_err = 0.0;
    let mut class_ok = 0usize;
    let mut prediction_count = 0usize;
    let mut precision_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut f1_sum = 0.0;
    let mut hit_sum = 0.0;
    let mut ndcg_sum = 0.0;
    let mut genre_diversity_sum = 0.0;
    let mut user_count = 0usize;
    let mut recommended_items = HashSet::new();
    let mut popularity_sum = 0.0;
    let mut recommendation_total = 0usize;

    for rating in &test {
        if let Some(score) = recommender.score(rating.user_id, rating.movie_id) {
            let predicted = norm_to_rating(score.normalized_score);
            let err = (predicted - rating.rating).abs();
            abs_err += err;
            sq_err += err * err;
            class_ok += ((predicted >= 4.0) == (rating.rating >= 4.0)) as usize;
            prediction_count += 1;
        }
    }

    for user_id in users {
        let relevant = test
            .iter()
            .filter(|r| r.user_id == user_id && r.rating >= 4.0)
            .map(|r| r.movie_id)
            .collect::<HashSet<_>>();
        if relevant.is_empty() {
            continue;
        }
        let recs = recommender.recommend(user_id, top_n);
        let precision = precision_at_n(&recs, &relevant, top_n);
        let recall = recall_at_n(&recs, &relevant, top_n);
        let hits = hit_count_at_n(&recs, &relevant, top_n);
        for rec in recs.iter().take(top_n) {
            recommended_items.insert(rec.movie_id);
            popularity_sum += model
                .by_item
                .get(&rec.movie_id)
                .map(|ratings| ratings.len() as f32)
                .unwrap_or(0.0);
            recommendation_total += 1;
        }
        precision_sum += precision;
        recall_sum += recall;
        f1_sum += f1(precision, recall);
        hit_sum += if hits > 0 { 1.0 } else { 0.0 };
        ndcg_sum += ndcg_at_n(&recs, &relevant, top_n);
        genre_diversity_sum += genre_diversity_at_n(&recs, &model.dataset, top_n);
        user_count += 1;
    }

    let denom = prediction_count.max(1) as f32;
    let user_denom = user_count.max(1) as f32;
    let precision = precision_sum / user_denom;
    let recall = recall_sum / user_denom;
    EvaluationReport {
        prediction_coverage: prediction_count as f32 / test.len().max(1) as f32,
        prediction_count,
        ranking_user_count: user_count,
        accuracy: class_ok as f32 / denom,
        ndcg_at_n: ndcg_sum / user_denom,
        mae: abs_err / denom,
        rmse: (sq_err / denom).sqrt(),
        precision_at_n: precision,
        recall_at_n: recall,
        f1_at_n: f1_sum / user_denom,
        hit_rate_at_n: hit_sum / user_denom,
        catalog_coverage: recommended_items.len() as f32 / dataset.movies.len().max(1) as f32,
        avg_recommendation_popularity: popularity_sum / recommendation_total.max(1) as f32,
        avg_genre_diversity: genre_diversity_sum / user_denom,
    }
}

pub fn precision_at_n(recs: &[Recommendation], relevant: &HashSet<ItemId>, top_n: usize) -> f32 {
    let hits = recs
        .iter()
        .take(top_n)
        .filter(|rec| relevant.contains(&rec.movie_id))
        .count();
    hits as f32 / top_n.max(1) as f32
}

pub fn recall_at_n(recs: &[Recommendation], relevant: &HashSet<ItemId>, top_n: usize) -> f32 {
    if relevant.is_empty() {
        return 0.0;
    }
    hit_count_at_n(recs, relevant, top_n) as f32 / relevant.len() as f32
}

pub fn hit_rate_at_n(recs: &[Recommendation], relevant: &HashSet<ItemId>, top_n: usize) -> f32 {
    if hit_count_at_n(recs, relevant, top_n) > 0 {
        1.0
    } else {
        0.0
    }
}

pub fn f1(precision: f32, recall: f32) -> f32 {
    if precision + recall <= f32::EPSILON {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn hit_count_at_n(recs: &[Recommendation], relevant: &HashSet<ItemId>, top_n: usize) -> usize {
    recs.iter()
        .take(top_n)
        .filter(|rec| relevant.contains(&rec.movie_id))
        .count()
}

pub fn ndcg_at_n(recs: &[Recommendation], relevant: &HashSet<ItemId>, top_n: usize) -> f32 {
    let mut dcg = 0.0;
    for (idx, rec) in recs.iter().take(top_n).enumerate() {
        if relevant.contains(&rec.movie_id) {
            dcg += 1.0 / ((idx + 2) as f32).log2();
        }
    }
    let ideal = relevant.len().min(top_n);
    let idcg = (0..ideal)
        .map(|idx| 1.0 / ((idx + 2) as f32).log2())
        .sum::<f32>();
    if idcg <= f32::EPSILON {
        0.0
    } else {
        dcg / idcg
    }
}

fn genre_diversity_at_n(recs: &[Recommendation], dataset: &Dataset, top_n: usize) -> f32 {
    let mut genres = HashSet::new();
    let mut genre_slots = 0usize;
    for rec in recs.iter().take(top_n) {
        if let Some(movie) = dataset.movies.get(&rec.movie_id) {
            for genre in &movie.genres {
                genres.insert(genre);
                genre_slots += 1;
            }
        }
    }
    if genre_slots == 0 {
        0.0
    } else {
        genres.len() as f32 / genre_slots as f32
    }
}
