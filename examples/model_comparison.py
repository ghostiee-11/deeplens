"""DeepLens model comparison — pit LogisticRegression against RandomForest."""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import deeplens

# Prepare data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42,
)

# Train two models
lr = LogisticRegression(max_iter=300).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)

# Compare them side-by-side in the Model Arena
deeplens.compare(lr, rf, X_test, y_test)
