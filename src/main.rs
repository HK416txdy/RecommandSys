use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use recommand_sys::{
    DataModel, build_recommender, generate_report, generate_report_for_algorithms, load_dataset,
    prepare_data, search_movies,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        print_help();
        return Ok(());
    }
    match args[0].as_str() {
        "prepare" => {
            let data_dir =
                value_arg(&args, "--data-dir").unwrap_or_else(|| "data/ml-100k".to_string());
            prepare_data(Path::new(&data_dir)).map_err(|e| e.to_string())?;
            println!("prepared MovieLens-compatible data at {data_dir}");
        }
        "search" => {
            let data_dir =
                value_arg(&args, "--data-dir").unwrap_or_else(|| "data/ml-100k".to_string());
            ensure_data(&data_dir)?;
            let model = load_model(&data_dir)?;
            let query = value_arg(&args, "--query");
            let genre = value_arg(&args, "--genre");
            let year_from = value_arg(&args, "--year-from").and_then(|v| v.parse::<u16>().ok());
            let top_n = usize_arg(&args, "--top-n", 10);
            for movie in search_movies(&model, query.as_deref(), genre.as_deref(), year_from, top_n)
            {
                println!(
                    "{}\t{}\t{}\t{}",
                    movie.id,
                    movie.title,
                    movie
                        .release_year
                        .map(|y| y.to_string())
                        .unwrap_or_else(|| "-".to_string()),
                    movie.genres.join(",")
                );
            }
        }
        "recommend" => {
            let data_dir =
                value_arg(&args, "--data-dir").unwrap_or_else(|| "data/ml-100k".to_string());
            ensure_data(&data_dir)?;
            let model = load_model(&data_dir)?;
            let user_id = value_arg(&args, "--user-id")
                .ok_or("--user-id is required")?
                .parse::<u32>()
                .map_err(|_| "--user-id must be a number")?;
            let algorithm = value_arg(&args, "--algorithm").unwrap_or_else(|| "hybrid".to_string());
            let top_n = usize_arg(&args, "--top-n", 10);
            let recommender = build_recommender(&algorithm, &model);
            println!("algorithm={algorithm}, user_id={user_id}, top_n={top_n}");
            for rec in recommender.recommend(user_id, top_n) {
                if let Some(w) = rec.weights {
                    println!(
                        "{}\t{}\tscore={:.4}\tpred={:.2}\t{}\tweights[c={:.2},k={:.2},ucf={:.2},icf={:.2},pop={:.2},mf={:.2}]",
                        rec.movie_id,
                        rec.title,
                        rec.score,
                        rec.predicted_rating,
                        rec.reason,
                        w.content,
                        w.knowledge,
                        w.user_cf,
                        w.item_cf,
                        w.popularity,
                        w.matrix
                    );
                } else {
                    println!(
                        "{}\t{}\tscore={:.4}\tpred={:.2}\t{}",
                        rec.movie_id, rec.title, rec.score, rec.predicted_rating, rec.reason
                    );
                }
            }
        }
        "evaluate" => {
            let data_dir =
                value_arg(&args, "--data-dir").unwrap_or_else(|| "data/ml-100k".to_string());
            ensure_data(&data_dir)?;
            let dataset = load_dataset(Path::new(&data_dir)).map_err(|e| e.to_string())?;
            let top_n = usize_arg(&args, "--top-n", 10);
            let holdout_ratio = value_arg(&args, "--holdout-ratio")
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.2);
            let report_path = value_arg(&args, "--report")
                .unwrap_or_else(|| "reports/test_report.md".to_string());
            let algorithm = value_arg(&args, "--algorithm").unwrap_or_else(|| "all".to_string());
            let report = if algorithm == "all" {
                generate_report(&dataset, top_n, holdout_ratio)
            } else {
                generate_report_for_algorithms(&dataset, top_n, holdout_ratio, &[&algorithm])
            };
            let path = PathBuf::from(&report_path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::write(&path, report).map_err(|e| e.to_string())?;
            println!("wrote report to {}", path.display());
        }
        _ => print_help(),
    }
    Ok(())
}

fn ensure_data(data_dir: &str) -> Result<(), String> {
    let path = Path::new(data_dir);
    if !path.join("u.data").exists() || !path.join("u.item").exists() {
        prepare_data(path).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn load_model(data_dir: &str) -> Result<DataModel, String> {
    let dataset = load_dataset(Path::new(data_dir)).map_err(|e| e.to_string())?;
    Ok(DataModel::new(dataset))
}

fn value_arg(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn usize_arg(args: &[String], name: &str, default: usize) -> usize {
    value_arg(args, name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn print_help() {
    println!(
        "Rust search and dynamic hybrid recommender\n\n\
commands:\n\
  prepare --data-dir data/ml-100k\n\
  search --query \"Star Wars\" --genre Action --year-from 1970 --top-n 10\n\
  recommend --user-id 196 --algorithm hybrid --top-n 10\n\
  evaluate --algorithm all --top-n 10 --holdout-ratio 0.2 --report reports/test_report.md\n\n\
algorithms: content, knowledge, user-cf, item-cf, popularity, matrix, hybrid"
    );
}
