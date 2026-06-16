use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use crate::types::{Dataset, GENRES, ItemId, Movie, Rating, UserId};

#[derive(Clone, Debug)]
pub struct DataModel {
    pub dataset: Dataset,
    pub by_user: HashMap<UserId, Vec<Rating>>,
    pub by_item: HashMap<ItemId, Vec<Rating>>,
    pub user_mean: HashMap<UserId, f32>,
    pub item_mean: HashMap<ItemId, f32>,
    pub global_mean: f32,
}

impl DataModel {
    pub fn new(dataset: Dataset) -> Self {
        let mut by_user: HashMap<UserId, Vec<Rating>> = HashMap::new();
        let mut by_item: HashMap<ItemId, Vec<Rating>> = HashMap::new();
        let mut sum = 0.0;
        for rating in &dataset.ratings {
            by_user
                .entry(rating.user_id)
                .or_default()
                .push(rating.clone());
            by_item
                .entry(rating.movie_id)
                .or_default()
                .push(rating.clone());
            sum += rating.rating;
        }
        let global_mean = if dataset.ratings.is_empty() {
            3.0
        } else {
            sum / dataset.ratings.len() as f32
        };
        let user_mean = by_user
            .iter()
            .map(|(user_id, ratings)| {
                let mean = ratings.iter().map(|r| r.rating).sum::<f32>() / ratings.len() as f32;
                (*user_id, mean)
            })
            .collect();
        let item_mean = by_item
            .iter()
            .map(|(item_id, ratings)| {
                let mean = ratings.iter().map(|r| r.rating).sum::<f32>() / ratings.len() as f32;
                (*item_id, mean)
            })
            .collect();
        Self {
            dataset,
            by_user,
            by_item,
            user_mean,
            item_mean,
            global_mean,
        }
    }

    pub fn user_seen(&self, user_id: UserId) -> HashSet<ItemId> {
        self.by_user
            .get(&user_id)
            .into_iter()
            .flatten()
            .map(|r| r.movie_id)
            .collect()
    }

    pub fn movie_title(&self, item_id: ItemId) -> String {
        self.dataset
            .movies
            .get(&item_id)
            .map(|m| m.title.clone())
            .unwrap_or_else(|| format!("Movie #{item_id}"))
    }
}

pub fn prepare_data(data_dir: &Path) -> io::Result<()> {
    fs::create_dir_all(data_dir)?;
    let item_path = data_dir.join("u.item");
    let data_path = data_dir.join("u.data");
    if item_path.exists() && data_path.exists() {
        return Ok(());
    }

    let items = [
        (
            1,
            "Toy Story (1995)",
            1995,
            vec!["Animation", "Children", "Comedy"],
        ),
        (
            2,
            "GoldenEye (1995)",
            1995,
            vec!["Action", "Adventure", "Thriller"],
        ),
        (3, "Four Rooms (1995)", 1995, vec!["Comedy"]),
        (4, "Get Shorty (1995)", 1995, vec!["Comedy", "Crime"]),
        (
            5,
            "Copycat (1995)",
            1995,
            vec!["Crime", "Drama", "Thriller"],
        ),
        (6, "Twelve Monkeys (1995)", 1995, vec!["Sci-Fi", "Thriller"]),
        (7, "Babe (1995)", 1995, vec!["Children", "Comedy", "Drama"]),
        (8, "Richard III (1995)", 1995, vec!["Drama", "War"]),
        (
            9,
            "Star Wars (1977)",
            1977,
            vec!["Action", "Adventure", "Sci-Fi", "War"],
        ),
        (10, "Pulp Fiction (1994)", 1994, vec!["Crime", "Drama"]),
        (
            11,
            "The Matrix (1999)",
            1999,
            vec!["Action", "Sci-Fi", "Thriller"],
        ),
        (
            12,
            "Sense and Sensibility (1995)",
            1995,
            vec!["Drama", "Romance"],
        ),
    ];

    let mut item_file = fs::File::create(item_path)?;
    for (id, title, year, genres) in items {
        let mut flags = vec!["0"; GENRES.len()];
        for genre in genres {
            if let Some(idx) = GENRES.iter().position(|g| *g == genre) {
                flags[idx] = "1";
            }
        }
        writeln!(
            item_file,
            "{}|{}|01-Jan-{}|unknown|http://example.invalid|{}",
            id,
            title,
            year,
            flags.join("|")
        )?;
    }

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
    let mut data_file = fs::File::create(data_path)?;
    for (idx, (user, item, rating)) in ratings.iter().enumerate() {
        writeln!(
            data_file,
            "{user}\t{item}\t{rating}\t{}",
            874_965_000 + idx as u64
        )?;
    }
    Ok(())
}

pub fn load_dataset(data_dir: &Path) -> io::Result<Dataset> {
    let item_text = String::from_utf8_lossy(&fs::read(data_dir.join("u.item"))?).into_owned();
    let data_text = String::from_utf8_lossy(&fs::read(data_dir.join("u.data"))?).into_owned();

    let mut movies = HashMap::new();
    for line in item_text.lines().filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split('|').collect();
        if fields.len() < 24 {
            continue;
        }
        let Ok(id) = fields[0].parse::<ItemId>() else {
            continue;
        };
        let title = fields[1].to_string();
        let release_year = extract_year(&title).or_else(|| extract_year(fields[2]));
        let genres = fields[5..]
            .iter()
            .enumerate()
            .take(GENRES.len())
            .filter(|(_, flag)| **flag == "1")
            .map(|(idx, _)| GENRES[idx].to_string())
            .collect();
        movies.insert(
            id,
            Movie {
                id,
                title,
                genres,
                release_year,
            },
        );
    }

    let mut ratings = Vec::new();
    for line in data_text.lines().filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            continue;
        }
        let (Ok(user_id), Ok(movie_id), Ok(rating), Ok(timestamp)) = (
            fields[0].parse(),
            fields[1].parse(),
            fields[2].parse(),
            fields[3].parse(),
        ) else {
            continue;
        };
        ratings.push(Rating {
            user_id,
            movie_id,
            rating,
            timestamp,
        });
    }
    Ok(Dataset { movies, ratings })
}

fn extract_year(text: &str) -> Option<u16> {
    for i in 0..text.len().saturating_sub(3) {
        let year_text = text.get(i..i + 4)?;
        if year_text.chars().all(|c| c.is_ascii_digit()) {
            let year: u16 = year_text.parse().ok()?;
            if (1900..=2100).contains(&year) {
                return Some(year);
            }
        }
    }
    None
}
