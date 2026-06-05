use std::collections::HashMap;

pub type UserId = u32;
pub type ItemId = u32;

pub const GENRES: [&str; 19] = [
    "Unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
];

#[derive(Clone, Debug)]
pub struct Movie {
    pub id: ItemId,
    pub title: String,
    pub genres: Vec<String>,
    pub release_year: Option<u16>,
}

#[derive(Clone, Debug)]
pub struct Rating {
    pub user_id: UserId,
    pub movie_id: ItemId,
    pub rating: f32,
    pub timestamp: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Dataset {
    pub movies: HashMap<ItemId, Movie>,
    pub ratings: Vec<Rating>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AlgorithmWeights {
    pub content: f32,
    pub knowledge: f32,
    pub user_cf: f32,
    pub item_cf: f32,
    pub popularity: f32,
    pub matrix: f32,
}

impl AlgorithmWeights {
    pub fn sum(self) -> f32 {
        self.content + self.knowledge + self.user_cf + self.item_cf + self.popularity + self.matrix
    }

    pub fn normalize(self) -> Self {
        let sum = self.sum();
        if sum <= f32::EPSILON {
            return Self {
                content: 0.22,
                knowledge: 0.18,
                user_cf: 0.18,
                item_cf: 0.18,
                popularity: 0.10,
                matrix: 0.14,
            };
        }
        Self {
            content: self.content / sum,
            knowledge: self.knowledge / sum,
            user_cf: self.user_cf / sum,
            item_cf: self.item_cf / sum,
            popularity: self.popularity / sum,
            matrix: self.matrix / sum,
        }
    }

    pub fn dominant(self) -> &'static str {
        let pairs = [
            ("content", self.content),
            ("knowledge", self.knowledge),
            ("user-cf", self.user_cf),
            ("item-cf", self.item_cf),
            ("popularity", self.popularity),
            ("matrix", self.matrix),
        ];
        pairs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|p| p.0)
            .unwrap_or("hybrid")
    }
}

#[derive(Clone, Debug)]
pub struct ScoredRecommendation {
    pub raw_score: f32,
    pub normalized_score: f32,
    pub confidence: f32,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct Recommendation {
    pub movie_id: ItemId,
    pub title: String,
    pub score: f32,
    pub predicted_rating: f32,
    pub reason: String,
    pub weights: Option<AlgorithmWeights>,
}
