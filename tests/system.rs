use std::collections::HashSet;
use std::fs;

use recommand_sys::blending::HybridRecommender;
use recommand_sys::eval::{f1, hit_rate_at_n, ndcg_at_n, precision_at_n, recall_at_n};
use recommand_sys::recommender::{
    ContentRecommender, KnowledgeRecommender, Recommender, build_recommender,
};
use recommand_sys::{
    DataModel, Recommendation, generate_report, generate_report_for_algorithms, load_dataset,
    search_movies,
};

fn sample_model(name: &str) -> DataModel {
    let dir = std::env::temp_dir().join(format!("recommand_sys_test_data_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    write_sample_data(&dir);
    DataModel::new(load_dataset(&dir).unwrap())
}

fn write_sample_data(dir: &std::path::Path) {
    let items = [
        (
            1,
            "Toy Story (1995)",
            1995,
            ["Animation", "Children", "Comedy"].as_slice(),
        ),
        (
            2,
            "GoldenEye (1995)",
            1995,
            ["Action", "Adventure", "Thriller"].as_slice(),
        ),
        (3, "Four Rooms (1995)", 1995, ["Comedy"].as_slice()),
        (4, "Get Shorty (1995)", 1995, ["Comedy", "Crime"].as_slice()),
        (
            5,
            "Copycat (1995)",
            1995,
            ["Crime", "Drama", "Thriller"].as_slice(),
        ),
        (
            6,
            "Twelve Monkeys (1995)",
            1995,
            ["Sci-Fi", "Thriller"].as_slice(),
        ),
        (
            7,
            "Babe (1995)",
            1995,
            ["Children", "Comedy", "Drama"].as_slice(),
        ),
        (8, "Richard III (1995)", 1995, ["Drama", "War"].as_slice()),
        (
            9,
            "Star Wars (1977)",
            1977,
            ["Action", "Adventure", "Sci-Fi", "War"].as_slice(),
        ),
        (
            10,
            "Pulp Fiction (1994)",
            1994,
            ["Crime", "Drama"].as_slice(),
        ),
        (
            11,
            "The Matrix (1999)",
            1999,
            ["Action", "Sci-Fi", "Thriller"].as_slice(),
        ),
        (
            12,
            "Sense and Sensibility (1995)",
            1995,
            ["Drama", "Romance"].as_slice(),
        ),
    ];
    let mut item_text = String::new();
    for (id, title, year, genres) in items {
        let mut flags = vec!["0"; recommand_sys::GENRES.len()];
        for genre in genres {
            if let Some(idx) = recommand_sys::GENRES.iter().position(|g| g == genre) {
                flags[idx] = "1";
            }
        }
        item_text.push_str(&format!(
            "{}|{}|01-Jan-{}|unknown|http://example.invalid|{}\n",
            id,
            title,
            year,
            flags.join("|")
        ));
    }
    fs::write(dir.join("u.item"), item_text).unwrap();

    let ratings = [
        (196, 1, 5),
        (196, 6, 4),
        (196, 9, 5),
        (196, 11, 5),
        (196, 12, 2),
        (1, 1, 5),
        (1, 3, 4),
        (1, 4, 4),
        (1, 7, 5),
        (1, 10, 3),
        (2, 2, 4),
        (2, 5, 5),
        (2, 6, 4),
        (2, 9, 5),
        (2, 11, 5),
        (3, 8, 5),
        (3, 10, 5),
        (3, 12, 5),
        (3, 5, 3),
        (3, 4, 2),
        (4, 1, 4),
        (4, 7, 4),
        (4, 12, 5),
        (4, 3, 3),
        (4, 8, 4),
        (5, 2, 5),
        (5, 9, 5),
        (5, 11, 4),
        (5, 6, 5),
        (5, 5, 2),
        (6, 10, 5),
        (6, 4, 4),
        (6, 5, 4),
        (6, 12, 3),
        (6, 8, 4),
    ];
    let data_text = ratings
        .iter()
        .enumerate()
        .map(|(idx, (user, item, rating))| {
            format!("{user}\t{item}\t{rating}\t{}\n", 874_965_000 + idx as u64)
        })
        .collect::<String>();
    fs::write(dir.join("u.data"), data_text).unwrap();
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
    assert!(report.contains("预测覆盖率"));
    assert!(report.contains("方法与数学建模"));
    assert!(report.contains("Precision@N"));
    assert!(report.contains("Recall@N"));
    assert!(report.contains("F1@N"));
    assert!(report.contains("HitRate@N"));
    assert!(report.contains("nDCG@N"));
    assert!(report.contains("目录覆盖率"));
    assert!(report.contains("平均推荐流行度"));
    assert!(report.contains("类型多样性"));
    assert!(report.contains("结果分析"));
    assert!(report.contains("## 局限性"));
}

#[test]
fn report_can_be_limited_to_selected_algorithms() {
    let model = sample_model("single_report");
    let report = generate_report_for_algorithms(&model.dataset, 5, 0.2, &["content"]);
    assert!(report.contains("| content |"));
    assert!(!report.contains("| hybrid |"));
}
