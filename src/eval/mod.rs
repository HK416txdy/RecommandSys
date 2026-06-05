use std::collections::{HashMap, HashSet};

use crate::data::DataModel;
use crate::math::norm_to_rating;
use crate::recommender::build_recommender;
use crate::types::{Dataset, ItemId, Rating, Recommendation, UserId};

#[derive(Clone, Debug, Default)]
pub struct EvaluationReport {
    pub accuracy: f32,
    pub ndcg_at_n: f32,
    pub mae: f32,
    pub rmse: f32,
    pub precision_at_n: f32,
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
    let mut ndcg_sum = 0.0;
    let mut user_count = 0usize;

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
        precision_sum += precision_at_n(&recs, &relevant, top_n);
        ndcg_sum += ndcg_at_n(&recs, &relevant, top_n);
        user_count += 1;
    }

    let denom = prediction_count.max(1) as f32;
    let user_denom = user_count.max(1) as f32;
    EvaluationReport {
        accuracy: class_ok as f32 / denom,
        ndcg_at_n: ndcg_sum / user_denom,
        mae: abs_err / denom,
        rmse: (sq_err / denom).sqrt(),
        precision_at_n: precision_sum / user_denom,
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
