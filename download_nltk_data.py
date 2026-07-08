"""
Run once before training if you hit NLTK lookup errors:
    python src/download_nltk_data.py

PorterStemmer itself doesn't need external data, but this is kept
as a placeholder in case you extend the pipeline with tokenizers,
stopword lists, or lemmatizers (e.g. wordnet) that do.
"""
import nltk

for pkg in ["punkt", "stopwords", "wordnet"]:
    nltk.download(pkg)
