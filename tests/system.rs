use std::collections::HashSet;
use std::fs;

use recommand_sys::blending::HybridRecommender;
use recommand_sys::eval::{f1, hit_rate_at_n, ndcg_at_n, precision_at_n, recall_at_n};
use recommand_sys::recommender::{
    ContentRecommender, KnowledgeRecommender, Recommender, build_recommender,
};
use recommand_sys::{
    DataModel, Recommendation, generate_report, generate_report_for_algorithms, load_dataset,
    prepare_data, search_movies,
};

fn sample_model(name: &str) -> DataModel {
    let dir = std::env::temp_dir().join(format!("recommand_sys_test_data_{name}"));
    let _ = fs::remove_dir_all(&dir);
    prepare_data(&dir).unwrap();
    DataModel::new(load_dataset(&dir).unwrap())
}

#[test]
fn loads_movielens_compatible_data() {
    let model = sample_model("loads");
    assert!(model.dataset.movies.len() >= 10);
    assert!(model.dataset.ratings.len() >= 30);
    assert!(
        model
            .dataset
            .movies
            .get(&9)
            .unwrap()
            .genres
            .contains(&"Sci-Fi".to_string())
    );
}

#[test]
fn search_filters_query_genre_and_year() {
    let model = sample_model("search");
    let results = search_movies(&model, Some("Star"), Some("Action"), Some(1970), 5);
    assert_eq!(results[0].title, "Star Wars (1977)");
}

#[test]
fn metric_helpers_match_known_values() {
    let recs = vec![
        Recommendation {
            movie_id: 1,
            title: "A".to_string(),
            score: 1.0,
            predicted_rating: 5.0,
            reason: String::new(),
            weights: None,
        },
        Recommendation {
            movie_id: 2,
            title: "B".to_string(),
            score: 0.9,
            predicted_rating: 4.6,
            reason: String::new(),
            weights: None,
        },
    ];
    let relevant = HashSet::from([1, 3]);
    assert!((precision_at_n(&recs, &relevant, 2) - 0.5).abs() < 0.001);
    assert!((recall_at_n(&recs, &relevant, 2) - 0.5).abs() < 0.001);
    assert!((hit_rate_at_n(&recs, &relevant, 2) - 1.0).abs() < 0.001);
    assert!((f1(0.5, 0.5) - 0.5).abs() < 0.001);
    assert!((ndcg_at_n(&recs, &HashSet::from([1]), 2) - 1.0).abs() < 0.001);
}

#[test]
fn content_recommendation_skips_seen_items() {
    let model = sample_model("content");
    let recommender = ContentRecommender::new(&model);
    let seen = model.user_seen(196);
    let results = recommender.recommend(196, 5);
    assert!(!results.is_empty());
    assert!(results.iter().all(|rec| !seen.contains(&rec.movie_id)));
}

#[test]
fn knowledge_rules_affect_score() {
    let model = sample_model("knowledge");
    let recommender = KnowledgeRecommender::new(&model);
    let action = recommender.score(196, 2).unwrap().normalized_score;
    let romance = recommender.score(196, 12).unwrap().normalized_score;
    assert!(action >= romance);
}

#[test]
fn single_algorithms_return_finite_scores() {
    let model = sample_model("single_algorithms");
    for name in ["user-cf", "item-cf", "popularity", "matrix"] {
        let recommender = build_recommender(name, &model);
        let score = recommender.score(196, 2).unwrap();
        assert!(score.normalized_score.is_finite());
        assert!((0.0..=1.0).contains(&score.normalized_score));
    }
}

#[test]
fn hybrid_dynamic_weights_are_normalized() {
    let model = sample_model("hybrid");
    let recs = HybridRecommender::new(&model).recommend(196, 3);
    assert!(!recs.is_empty());
    for rec in recs {
        let weights = rec.weights.unwrap();
        assert!((weights.sum() - 1.0).abs() < 0.001);
        assert!(rec.reason.contains("dynamic hybrid"));
    }
}

#[test]
fn report_contains_required_metric_names_and_analysis_sections() {
    let model = sample_model("report");
    let report = generate_report(&model.dataset, 5, 0.2);
    assert!(report.contains("分类准确度"));
    assert!(report.contains("Precision@N"));
    assert!(report.contains("Recall@N"));
    assert!(report.contains("F1@N"));
    assert!(report.contains("HitRate@N"));
    assert!(report.contains("目录覆盖率"));
    assert!(report.contains("平均推荐流行度"));
    assert!(report.contains("类型多样性"));
    assert!(report.contains("结果分析"));
    assert!(report.contains("系统能力与局限性"));
}

#[test]
fn report_can_be_limited_to_selected_algorithms() {
    let model = sample_model("single_report");
    let report = generate_report_for_algorithms(&model.dataset, 5, 0.2, &["content"]);
    assert!(report.contains("| content |"));
    assert!(!report.contains("| hybrid |"));
}
