"""DeepLens text embeddings — explore 20newsgroups with TF-IDF + PCA."""
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

import deeplens

# Load a small subset for speed
news = fetch_20newsgroups(subset="train", categories=None, random_state=42)
texts = news.data[:500]
labels = [news.target_names[t] for t in news.target[:500]]

df = pd.DataFrame({"text": texts, "label": labels})

# One-liner: embed text, reduce to 2-D, and open the explorer
deeplens.explore(df, text_col="text", label_col="label")
