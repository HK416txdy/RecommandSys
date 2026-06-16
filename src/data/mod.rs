use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
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

pub fn ensure_dataset_files(data_dir: &Path) -> io::Result<()> {
    let item_path = data_dir.join("u.item");
    let data_path = data_dir.join("u.data");
    if !item_path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("missing MovieLens item file: {}", item_path.display()),
        ));
    }
    if !data_path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("missing MovieLens ratings file: {}", data_path.display()),
        ));
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
