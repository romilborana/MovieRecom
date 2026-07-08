"""
test_train.py
--------------
Unit tests for the feature-engineering + recommendation logic in src/train.py.

Run:
    pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import (  # noqa: E402
    build_tags,
    clean_overview,
    collapse,
    fetch_crew,
    parse_names,
    recommend,
    stem_tokens,
)


# ──────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────
def test_parse_names_extracts_name_field():
    text = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
    assert parse_names(text) == ["Action", "Adventure"]


def test_parse_names_respects_limit():
    text = '[{"id": 1, "name": "A"}, {"id": 2, "name": "B"}, {"id": 3, "name": "C"}]'
    assert parse_names(text, limit=2) == ["A", "B"]


def test_parse_names_handles_malformed_input():
    assert parse_names("not a valid list") == []


def test_fetch_crew_filters_by_job():
    text = (
        '[{"name": "Jane Doe", "job": "Director"}, '
        '{"name": "John Roe", "job": "Producer"}]'
    )
    assert fetch_crew(text) == ["Jane Doe"]


def test_fetch_crew_multiple_jobs():
    text = (
        '[{"name": "Jane Doe", "job": "Director"}, '
        '{"name": "John Roe", "job": "Writer"}]'
    )
    assert set(fetch_crew(text, jobs=("Director", "Writer"))) == {"Jane Doe", "John Roe"}


def test_collapse_removes_spaces_and_lowercases():
    assert collapse(["Sam Worthington", "Zoe Saldana"]) == ["samworthington", "zoesaldana"]


def test_clean_overview_strips_punctuation_and_lowercases():
    tokens = clean_overview("A Marine, on Pandora! Fights back.")
    assert tokens == ["a", "marine", "on", "pandora", "fights", "back"]


def test_stem_tokens_collapses_related_words():
    assert stem_tokens(["running", "runs", "run"]) == ["run", "run", "run"]


# ──────────────────────────────────────────────────────────────────
# Tag building
# ──────────────────────────────────────────────────────────────────
def test_build_tags_applies_field_weights():
    row = pd.Series(
        {
            "overview_tokens": ["hero", "saves", "world"],
            "genres": ["action"],
            "keywords": ["hero"],
            "cast": ["someactor"],
            "director": ["somedirector"],
        }
    )
    tags = build_tags(row)
    # genres repeated 3x, director repeated 3x, keyword/cast repeated 2x
    assert tags.count("action") == 3
    assert tags.count("somedirector") == 3
    assert tags.count("someactor") == 2


# ──────────────────────────────────────────────────────────────────
# Recommend function
# ──────────────────────────────────────────────────────────────────
@pytest.fixture
def tiny_catalog():
    new = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "title": ["Alpha", "Beta", "Gamma"],
            "tags": ["a b c", "a b d", "x y z"],
        }
    )
    # Alpha & Beta deliberately similar; Gamma dissimilar
    similarity = np.array(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.05],
            [0.1, 0.05, 1.0],
        ]
    )
    return new, similarity


def test_recommend_returns_most_similar_first(tiny_catalog):
    new, similarity = tiny_catalog
    results = recommend(new, similarity, "Alpha", top_n=2)
    assert results == ["Beta", "Gamma"]


def test_recommend_handles_typo_with_fuzzy_match(tiny_catalog, caplog):
    new, similarity = tiny_catalog
    results = recommend(new, similarity, "Alfa", top_n=1)  # typo for "Alpha"
    assert results == ["Beta"]


def test_recommend_returns_empty_for_no_match(tiny_catalog):
    new, similarity = tiny_catalog
    results = recommend(new, similarity, "Completely Unrelated Title Zzzz", top_n=1)
    assert results == []
