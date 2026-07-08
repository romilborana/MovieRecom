"""
app.py
------
Streamlit front-end for the content-based movie recommender.

Run locally:
    streamlit run app.py

Requires:
    models/movie_list.pkl
    models/similarity.pkl
    A TMDB API key set as the TMDB_API_KEY environment variable
    (or in .streamlit/secrets.toml as TMDB_API_KEY = "...") for poster fetching.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

MODELS_DIR = Path("models")
PLACEHOLDER_POSTER = "https://via.placeholder.com/300x450.png?text=No+Poster"


# ──────────────────────────────────────────────────────────────────
# Cached loaders — Streamlit re-runs the whole script on every
# interaction, so caching prevents re-reading multi-MB pickles
# and re-hitting the network on every click.
# ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    movies = pickle.load(open(MODELS_DIR / "movie_list.pkl", "rb"))
    similarity = pickle.load(open(MODELS_DIR / "similarity.pkl", "rb"))
    return movies, similarity


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_poster(movie_id: int) -> str:
    """Fetch a poster URL from TMDB. Falls back to a placeholder on any failure."""
    api_key = os.environ.get("TMDB_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["TMDB_API_KEY"]
        except Exception:
            api_key = ""

    if not api_key:
        return PLACEHOLDER_POSTER

    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": api_key, "language": "en-US"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return PLACEHOLDER_POSTER
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.RequestException:
        return PLACEHOLDER_POSTER


def recommend(movies: pd.DataFrame, similarity, title: str, top_n: int = 5):
    index = movies[movies["title"] == title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    results = []
    for i, score in distances[1 : top_n + 1]:
        row = movies.iloc[i]
        results.append(
            {
                "title": row.title,
                "movie_id": int(row.movie_id),
                "score": round(float(score), 3),
                "poster": fetch_poster(int(row.movie_id)),
            }
        )
    return results


# ──────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    st.title("🎬 Content-Based Movie Recommender")
    st.caption(
        "TF-IDF + cosine similarity over overview, genres, keywords, cast, and director "
        "— trained on the TMDB 5000 dataset."
    )

    if not (MODELS_DIR / "movie_list.pkl").exists():
        st.error(
            "Model artifacts not found. Run `python src/train.py` first to generate "
            "`models/movie_list.pkl` and `models/similarity.pkl`."
        )
        st.stop()

    movies, similarity = load_model()

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_title = st.selectbox(
            "Pick a movie you like:",
            options=sorted(movies["title"].values),
            index=None,
            placeholder="Start typing a title...",
        )
    with col2:
        top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    if st.button("Recommend", type="primary", disabled=selected_title is None):
        with st.spinner("Finding similar movies..."):
            results = recommend(movies, similarity, selected_title, top_n=top_n)

        st.subheader(f"Because you liked *{selected_title}*")
        cols = st.columns(len(results))
        for col, rec in zip(cols, results):
            with col:
                st.image(rec["poster"], use_container_width=True)
                st.markdown(f"**{rec['title']}**")
                st.caption(f"similarity: {rec['score']}")

    with st.expander("How this works"):
        st.markdown(
            """
            1. Each movie's overview, genres, keywords, top-3 cast, and director are combined
               into a single "tag" string (genres/cast/director are repeated to weight them
               more heavily than plot-summary words).
            2. Overview words are stemmed (Porter Stemmer) so *action*, *acting*, *actions*
               collapse to one token.
            3. The tag corpus is vectorized with **TF-IDF** (top 5000 terms, English stop
               words removed).
            4. **Cosine similarity** between every pair of movies is precomputed once at
               train time.
            5. At query time we just look up the row for the selected movie and sort the
               other movies by similarity score — O(n log n), no re-vectorization needed.
            """
        )


if __name__ == "__main__":
    main()
