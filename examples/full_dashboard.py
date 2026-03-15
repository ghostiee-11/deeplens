"""DeepLens full dashboard — manually build state and launch every module."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deeplens.config import DeepLensState
from deeplens.dashboard.app import DeepLensDashboard

# Step 1 — Load data
iris = load_iris(as_frame=True)
df = iris.frame
feature_cols = iris.feature_names
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_cols], df["target"], test_size=0.3, random_state=42,
)

# Step 2 — Train a model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Step 3 — Compute embeddings (PCA on features)
embeddings_2d = PCA(n_components=2).fit_transform(df[feature_cols])

# Step 4 — Populate state with all fields
state = DeepLensState(
    df=df,
    dataset_name="iris",
    feature_columns=feature_cols,
    label_column="target",
    embeddings_2d=embeddings_2d,
    labels=df["target"].values,
    class_names=list(iris.target_names),
    trained_model=model,
    model_name="RandomForest",
    predictions=model.predict(df[feature_cols]),
    probabilities=model.predict_proba(df[feature_cols]),
)

# Step 5 — Launch the dashboard (all modules use the shared state)
app = DeepLensDashboard(state=state)
app.show(title="DeepLens — Full Dashboard")
