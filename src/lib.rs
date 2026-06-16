pub mod blending;
pub mod data;
pub mod eval;
pub mod math;
pub mod recommender;
pub mod report;
pub mod search;
pub mod types;

pub use data::{DataModel, ensure_dataset_files, load_dataset};
pub use eval::{EvaluationReport, evaluate_algorithm, split_holdout};
pub use recommender::{Recommender, build_recommender};
pub use report::{generate_report, generate_report_for_algorithms};
pub use search::search_movies;
pub use types::{
    AlgorithmWeights, Dataset, GENRES, ItemId, Movie, Rating, Recommendation, ScoredRecommendation,
    UserId,
};
