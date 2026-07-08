"""
train.py
--------
Builds a content-based movie recommender from the TMDB 5000 dataset.

Pipeline:
    1. Load & merge movies + credits
    2. Parse nested JSON-like columns (genres, keywords, cast, crew)
    3. Clean & stem text
    4. Build a weighted "tags" corpus per movie
    5. Vectorize with TF-IDF
    6. Compute cosine similarity matrix
    7. Evaluate with genre-overlap precision@k
    8. Persist model artifacts to models/

Run:
    python src/train.py --movies data/tmdb_5000_movies.csv --credits data/tmdb_5000_credits.csv
"""

from __future__ import annotations

import argparse
import ast
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from difflib import get_close_matches
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ps = PorterStemmer()

# Feature weights: how many times each field's tokens are repeated in the
# combined "tags" string. Repetition inflates a token's term-frequency,
# nudging TF-IDF toward treating that field as more important without
# needing a custom multi-field vectorizer.
GENRE_WEIGHT = 3
KEYWORD_WEIGHT = 2
DIRECTOR_WEIGHT = 3
CAST_WEIGHT = 2


# ──────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────
def parse_names(text: str, limit: int | None = None) -> list[str]:
    """Extract 'name' fields from a JSON-like stringified list of dicts."""
    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    names = [i["name"] for i in items]
    return names[:limit] if limit else names


def fetch_crew(text: str, jobs: tuple[str, ...] = ("Director",)) -> list[str]:
    """Extract crew members matching given job titles (default: Director)."""
    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return [i["name"] for i in items if i["job"] in jobs]


def collapse(tokens: list[str]) -> list[str]:
    """Remove internal spaces + lowercase so multi-word names are single tokens."""
    return [t.replace(" ", "").lower() for t in tokens]


def clean_overview(text: str) -> list[str]:
    """Lowercase, strip punctuation, tokenize."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return text.split()


def stem_tokens(tokens: list[str]) -> list[str]:
    return [ps.stem(t) for t in tokens]


# ──────────────────────────────────────────────────────────────────
# Pipeline stages
# ──────────────────────────────────────────────────────────────────
def load_data(movies_path: str, credits_path: str) -> pd.DataFrame:
    logger.info("Loading data from %s and %s", movies_path, credits_path)
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    movies = movies.merge(credits, on="title")

    movies = movies[[
        "movie_id", "title", "overview", "genres", "keywords",
        "cast", "crew", "vote_average", "vote_count", "popularity",
    ]]
    before = len(movies)
    movies.dropna(subset=["overview", "genres", "keywords", "cast", "crew"], inplace=True)
    logger.info("Dropped %d rows with missing values (%d remain)", before - len(movies), len(movies))
    return movies


def engineer_features(movies: pd.DataFrame) -> pd.DataFrame:
    logger.info("Parsing nested fields...")
    movies = movies.copy()
    movies["genres"] = movies["genres"].apply(parse_names)
    movies["keywords"] = movies["keywords"].apply(parse_names)
    movies["cast"] = movies["cast"].apply(lambda x: parse_names(x, limit=3))
    movies["director"] = movies["crew"].apply(fetch_crew)
    movies["overview_tokens"] = movies["overview"].apply(clean_overview)

    movies["genres"] = movies["genres"].apply(collapse)
    movies["keywords"] = movies["keywords"].apply(collapse)
    movies["cast"] = movies["cast"].apply(collapse)
    movies["director"] = movies["director"].apply(collapse)
    return movies


def build_tags(row: pd.Series) -> str:
    tokens = (
        stem_tokens(row["overview_tokens"])
        + row["genres"] * GENRE_WEIGHT
        + row["keywords"] * KEYWORD_WEIGHT
        + row["cast"] * CAST_WEIGHT
        + row["director"] * DIRECTOR_WEIGHT
    )
    return " ".join(tokens)


def vectorize(tags: pd.Series, max_features: int = 5000):
    logger.info("Fitting TF-IDF vectorizer (max_features=%d)...", max_features)
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    vectors = tfidf.fit_transform(tags)
    logger.info("Vector matrix shape: %s", vectors.shape)
    return tfidf, vectors


def evaluate_recommender(
    new: pd.DataFrame,
    genre_lookup: pd.Series,
    similarity: np.ndarray,
    sample_size: int = 100,
    k: int = 5,
    seed: int = 42,
) -> float:
    """
    Genre-overlap precision@k: for a random sample of movies, what fraction
    of the top-k recommendations share at least one genre with the query?
    This is a cheap, label-free proxy for recommendation relevance —
    there's no ground-truth "correct" recommendation list, so we use genre
    agreement as a sanity-checkable stand-in signal.
    """
    rng = np.random.default_rng(seed)
    n = min(sample_size, len(new))
    sample_idx = rng.choice(len(new), size=n, replace=False)

    precisions = []
    for idx in sample_idx:
        query_genres = set(genre_lookup[idx])
        distances = sorted(enumerate(similarity[idx]), reverse=True, key=lambda x: x[1])
        top_k_idx = [i for i, _ in distances[1 : k + 1]]
        hits = sum(1 for j in top_k_idx if genre_lookup[j] & query_genres)
        precisions.append(hits / k)

    return float(np.mean(precisions))


def recommend(new: pd.DataFrame, similarity: np.ndarray, movie: str, top_n: int = 5) -> list[str]:
    """CLI/debug helper — same logic the Streamlit app uses."""
    titles = new["title"].values
    if movie not in titles:
        close = get_close_matches(movie, titles, n=3, cutoff=0.6)
        if not close:
            logger.warning("'%s' not found and no close matches.", movie)
            return []
        logger.info("'%s' not found. Using closest match: '%s'", movie, close[0])
        movie = close[0]

    index = new[new["title"] == movie].index[0]
    distances = sorted(enumerate(similarity[index]), reverse=True, key=lambda x: x[1])
    return [new.iloc[i].title for i, _ in distances[1 : top_n + 1]]


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train the TMDB content-based recommender.")
    parser.add_argument("--movies", default="data/tmdb_5000_movies.csv")
    parser.add_argument("--credits", default="data/tmdb_5000_credits.csv")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--eval-sample-size", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    movies = load_data(args.movies, args.credits)
    movies = engineer_features(movies)

    logger.info("Building weighted tag corpus...")
    movies["tags"] = movies.apply(build_tags, axis=1)

    new = movies[["movie_id", "title", "tags", "vote_average", "vote_count", "popularity"]].reset_index(drop=True)
    genre_lookup = movies["genres"].reset_index(drop=True).apply(set)

    tfidf, vectors = vectorize(new["tags"], max_features=args.max_features)

    logger.info("Computing cosine similarity matrix...")
    similarity = cosine_similarity(vectors, vectors)

    score = evaluate_recommender(new, genre_lookup, similarity, sample_size=args.eval_sample_size)
    logger.info("Genre-overlap precision@5: %.3f", score)

    # quick sanity check on a known title
    sample_recs = recommend(new, similarity, "Gandhi")
    logger.info("Sample recommendations for 'Gandhi': %s", sample_recs)

    logger.info("Persisting artifacts to %s ...", out_dir)
    pickle.dump(new, open(out_dir / "movie_list.pkl", "wb"))
    pickle.dump(similarity, open(out_dir / "similarity.pkl", "wb"))
    pickle.dump(tfidf, open(out_dir / "vectorizer.pkl", "wb"))

    metrics = {"genre_overlap_precision_at_5": score, "n_movies": len(new), "vocab_size": vectors.shape[1]}
    with open(out_dir / "metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    logger.info("Done. Artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
