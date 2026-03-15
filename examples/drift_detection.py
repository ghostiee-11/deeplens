"""DeepLens drift detection — add noise to production data and detect drift."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

import deeplens

# Load reference (training) data
iris = load_iris(as_frame=True)
train_df = iris.frame.copy()

# Simulate production data with added Gaussian noise (drift)
rng = np.random.default_rng(42)
prod_df = iris.frame.copy()
feature_cols = iris.feature_names
prod_df[feature_cols] += rng.normal(loc=0.5, scale=0.3, size=prod_df[feature_cols].shape)

# Detect and visualize drift between the two distributions
deeplens.drift(train_df, prod_df)
