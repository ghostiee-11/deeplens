"""DeepLens image classification demo — CIFAR-style with sklearn on pixel features."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from deeplens.config import DeepLensState
from deeplens.dashboard.app import DeepLensDashboard

# Use sklearn digits as a lightweight "image" dataset (8x8 grayscale)
digits = load_digits()
df = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(64)])
df["label"] = digits.target

feature_cols = [c for c in df.columns if c.startswith("pixel_")]

# Train a classifier
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(df[feature_cols], df["label"])

# PCA for 2-D embedding
embeddings_2d = PCA(n_components=2).fit_transform(df[feature_cols])

# Build state
state = DeepLensState(
    df=df,
    dataset_name="digits",
    feature_columns=feature_cols,
    label_column="label",
    embeddings_2d=embeddings_2d,
    labels=np.array(df["label"].tolist()),
    class_names=[str(i) for i in range(10)],
    trained_model=model,
    model_name="RandomForest",
    predictions=model.predict(df[feature_cols]),
    probabilities=model.predict_proba(df[feature_cols]),
)

app = DeepLensDashboard(state=state)
app.show(title="DeepLens — Digits Classification")
