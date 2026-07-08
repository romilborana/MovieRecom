# 🎬 Content-Based Movie Recommender

A content-based recommendation system built on the TMDB 5000 dataset, served through
an interactive Streamlit app. Given a movie a user likes, it returns the most similar
titles by combining plot summary, genre, keywords, cast, and director into a single
TF-IDF vector space and ranking by cosine similarity.

**[Live demo](#deployment)** · **[How it works](#how-it-works)** · **[Results](#results)**

---

## Highlights

- **Content-based filtering** using TF-IDF + cosine similarity — no user interaction
  data required, so it works from a cold start.
- **Weighted multi-field feature engineering**: overview text, genres, keywords, top-3
  cast, and director are combined into one corpus, with structured fields (genre,
  director) up-weighted relative to free-text so metadata isn't drowned out by plot
  summary word counts.
- **Text normalization**: punctuation stripping, lowercasing, and Porter stemming to
  reduce vocabulary sparsity (e.g. *fighting/fights/fight* → one token).
- **Quantitative evaluation**: since there's no labeled ground truth for "correct"
  recommendations, the project uses a genre-overlap precision@k metric as a proxy for
  relevance, computed over a random sample of the catalog.
- **Production-shaped code**: training pipeline (`src/train.py`) is decoupled from the
  serving layer (`app.py`), with cached model loading, a fuzzy-match fallback for typos,
  unit tests (`pytest`), and pickled artifacts versioned separately from code.
- **Deployed UI**: Streamlit app with poster images pulled live from the TMDB API.

---

## Tech Stack

| Layer            | Tool                                   |
|-------------------|-----------------------------------------|
| Data processing   | pandas, numpy                          |
| NLP / vectorization | scikit-learn (`TfidfVectorizer`), NLTK (`PorterStemmer`) |
| Similarity        | scikit-learn (`cosine_similarity`)     |
| Serving           | Streamlit                              |
| External data     | TMDB API (posters)                     |
| Testing           | pytest                                 |

---

## Project Structure

```
movie-recommender/
├── app.py                        # Streamlit app (serving layer)
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example      # TMDB API key template
├── data/                         # raw CSVs (not committed — see Setup)
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── models/                       # generated artifacts (not committed)
│   ├── movie_list.pkl
│   ├── similarity.pkl
│   ├── vectorizer.pkl
│   └── metrics.txt
├── src/
│   ├── train.py                  # full training pipeline + CLI
│   └── download_nltk_data.py
├── tests/
│   └── test_train.py             # unit tests for feature engineering & recommend()
└── README.md
```

---

## Setup

```bash
git clone <your-repo-url>
cd movie-recommender
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset (TMDB 5000 Movie Dataset, e.g. from Kaggle) and place the two CSVs in `data/`:
- `data/tmdb_5000_movies.csv`
- `data/tmdb_5000_credits.csv`

## Train the model

```bash
python src/train.py --movies data/tmdb_5000_movies.csv --credits data/tmdb_5000_credits.csv
```

This writes `movie_list.pkl`, `similarity.pkl`, `vectorizer.pkl`, and `metrics.txt` to `models/`,
and logs the genre-overlap precision@5 score plus a sanity-check recommendation for "Gandhi".

## Run the app

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and add your TMDB API key (free: themoviedb.org/settings/api)
streamlit run app.py
```

## Run tests

```bash
pytest tests/ -v
```

---

## How It Works

1. **Merge & select** — `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` are merged on
   title; only fields relevant to content similarity are kept.
2. **Parse nested fields** — genres, keywords, cast, and crew arrive as stringified
   JSON lists of dicts (e.g. `[{"id": 28, "name": "Action"}]`); these are parsed with
   `ast.literal_eval` and reduced to plain name lists (top-3 cast, director only from crew).
3. **Normalize text** — overview text is lowercased, stripped of punctuation, and
   stemmed; all multi-word names/genres/keywords have internal spaces removed so
   "Science Fiction" becomes a single token `sciencefiction` instead of splitting into
   two unrelated words during vectorization.
4. **Weight fields** — genre and director tokens are repeated (3x), cast/keywords (2x),
   before being joined into one `tags` string per movie. This is a lightweight way to
   bias a single bag-of-words vectorizer toward structured metadata without building a
   more complex multi-field weighted vectorizer.
5. **Vectorize** — `TfidfVectorizer(max_features=5000, stop_words='english')` converts
   the tag corpus into a sparse term-frequency matrix, down-weighting common words that
   appear across many movies.
6. **Similarity** — pairwise `cosine_similarity` is computed once at train time and
   cached; at query time, recommending is just a similarity-matrix row lookup + sort.
7. **Serve** — the Streamlit app loads the pickled artifacts, lets the user pick a
   title (with a fuzzy-match fallback for typos), and renders the top-k results with
   posters fetched live from TMDB.

---

## Results

Run `python src/train.py` to regenerate `models/metrics.txt`. On the full TMDB 5000
dataset with default settings, this pipeline typically achieves:

- **~4800 movies** indexed after dropping incomplete rows
- **5000-term** TF-IDF vocabulary
- **Genre-overlap precision@5**: reported in `models/metrics.txt` after training —
  the fraction of top-5 recommendations sharing at least one genre with the query movie,
  averaged over a random sample. (Exact number depends on your local run; expect it in
  the 0.6–0.8 range given genre is a heavily-weighted feature.)

This metric is a proxy, not ground truth — two movies can be excellent recommendations
without sharing a genre (e.g. a thematically similar drama and biopic). It's included
to make the project's quality claims falsifiable rather than anecdotal.

---

## Possible Extensions

- Swap TF-IDF for sentence embeddings (e.g. `sentence-transformers`) on the overview
  field to capture semantic similarity beyond shared vocabulary.
- Add a popularity/rating tiebreaker so equally-similar movies rank by `vote_average`.
- Hybridize with collaborative filtering if user rating data becomes available.
- Cache TMDB poster lookups in a local SQLite DB to avoid repeated API calls across sessions.

---

## Deployment

The app is a standard Streamlit app and deploys directly to
[Streamlit Community Cloud](https://streamlit.io/cloud): connect the GitHub repo, set
`app.py` as the entry point, and add `TMDB_API_KEY` under app settings → Secrets.

---

## Dataset & Attribution

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
(Kaggle). This product uses the TMDB API but is not endorsed or certified by TMDB.
