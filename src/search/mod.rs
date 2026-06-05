use std::cmp::Ordering;

use crate::data::DataModel;
use crate::types::Movie;

pub fn search_movies(
    model: &DataModel,
    query: Option<&str>,
    genre: Option<&str>,
    year_from: Option<u16>,
    top_n: usize,
) -> Vec<Movie> {
    let query = query.unwrap_or("").to_lowercase();
    let genre = genre.map(|g| g.to_string());
    let mut scored = Vec::new();

    for movie in model.dataset.movies.values() {
        if let Some(ref genre) = genre
            && !movie.genres.iter().any(|g| g.eq_ignore_ascii_case(genre))
        {
            continue;
        }
        if let Some(year_from) = year_from
            && movie.release_year.unwrap_or(0) < year_from
        {
            continue;
        }

        let mut score = 0.0;
        let title = movie.title.to_lowercase();
        for token in query.split_whitespace() {
            if title.contains(token) {
                score += 2.0;
            }
        }
        score += model
            .by_item
            .get(&movie.id)
            .map(|ratings| ratings.len() as f32 * 0.01)
            .unwrap_or(0.0);
        scored.push((score, movie.id, movie.clone()));
    }

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    scored
        .into_iter()
        .take(top_n)
        .map(|(_, _, movie)| movie)
        .collect()
}
