"""
Figure generation for the paper's regression coefficient plots.

Loads pre-computed supply center delta data, fits a Ridge regression model,
computes bootstrap confidence intervals, and produces Figure 2 of the paper
(regression coefficients for advice settings and player experience predicting
supply center gains). Output is saved as results/regression_coefs.pdf.
"""

import json

from sklearn.linear_model import Ridge
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from scipy import stats
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_errorbar,
    theme_minimal,
    coord_flip,
    theme,
    element_text,
    element_blank,
    facet_grid,
)

# Load pre-computed feature matrix (X) and supply center deltas (Y)
with open("data/scs_delta_data.json", "r") as f:
    delta_scs = json.load(f)

X = delta_scs["X"]
Y = delta_scs["Y"]

X = np.array(X)
Y = np.array(Y)

# Standardize features for regularized regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute coefficient paths across a range of regularization strengths
coefs = []
alphas = np.logspace(-6, 6, 200)
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_scaled, Y)
    coefs.append(ridge.coef_)


def ridge_ci_bootstrap(X, y, alpha, n_bootstraps=10000, ci=95):
    """Compute bootstrap confidence intervals for Ridge regression coefficients."""
    n_samples, n_features = X.shape
    bootstrapped_coefs = np.zeros((n_bootstraps, n_features))

    for i in range(n_bootstraps):
        indices = resample(range(n_samples), n_samples=n_samples)
        X_resampled, y_resampled = X[indices], y[indices]

        model = Ridge(alpha=alpha)
        model.fit(X_resampled, y_resampled)

        bootstrapped_coefs[i] = model.coef_

    ci_lower = np.percentile(bootstrapped_coefs, (100 - ci) / 2, axis=0)
    ci_upper = np.percentile(bootstrapped_coefs, 100 - (100 - ci) / 2, axis=0)

    return ci_lower, ci_upper


# Cross-validation to select the best regularization parameter
alphase = np.logspace(-6, 6, 13)
ridge_cv = RidgeCV(alphas=alphase, store_cv_results=True)
ridge_cv.fit(X, Y)


def calculate_std_err(X, y):
    """Compute standard errors for Ridge regression coefficients."""
    model = Ridge()
    model.fit(X, y)

    y_pred = model.predict(X)

    residuals = y - y_pred
    n = X.shape[0]
    p = X.shape[1]
    df = n - p

    mse = np.sum(residuals**2) / df

    X_with_intercept = np.column_stack([np.ones(n), X])
    var_covar_mat = mse * np.linalg.inv(
        np.dot(X_with_intercept.T, X_with_intercept) + 1e-1 * np.eye(p + 1)
    )

    std_errs = np.sqrt(np.diag(var_covar_mat))

    return std_errs[1:]


# Feature names corresponding to the one-hot encoded columns in X:
# [0-6] Power, [7-9] Player type, [10] Phase count, [11-14] Advice setting
coef_names = [
    "AUSTRIA",
    "ENGLAND",
    "FRANCE",
    "GERMANY",
    "ITALY",
    "RUSSIA",
    "TURKEY",
    "novice",
    "experienced",
    "novice + advice",
    "#phases",
    "no advice",
    "message advice",
    "moves advice",
    "message + moves advice",
]

std_errs = calculate_std_err(X, Y)

# 95% confidence interval using t-distribution
t_value = stats.t.ppf(0.975, df=X.shape[0] - X.shape[1] - 1)

ci_lower = Ridge().fit(X_scaled, Y).coef_ - t_value * std_errs
ci_upper = Ridge().fit(X_scaled, Y).coef_ + t_value * std_errs


def plot_coef_with_ci(X, y, alpha):
    """
    Generate Figure 2: regression coefficients with bootstrap CIs.

    Plots player experience and advisor setting coefficients as a faceted
    dot-and-whisker plot, showing the effect of each feature on supply
    center gains per turn.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)

    coef = model.coef_

    ci_lower, ci_upper = ridge_ci_bootstrap(X, y, alpha, 10000, 95)
    category = ["player experience"] * 3 + ["advisor setting"] * 4

    df = pd.DataFrame(
        {
            "feature": coef_names[7:10] + coef_names[11:],
            "coefficient": np.concatenate([coef[7:10], coef[11:]]),
            "ci_lower": np.concatenate([ci_lower[7:10], ci_lower[11:]]),
            "ci_upper": np.concatenate([ci_upper[7:10], ci_upper[11:]]),
            "category": category,
        }
    )

    df["feature"] = pd.Categorical(df["feature"], categories=coef_names, ordered=True)
    df["category"] = pd.Categorical(
        df["category"],
        categories=["player experience", "advisor setting"],
        ordered=True,
    )

    plot = (
        ggplot(df, aes(x="feature", y="coefficient"))
        + geom_point(size=3)
        + geom_errorbar(aes(ymin="ci_lower", ymax="ci_upper"), width=0.2)
        + facet_grid("category ~ ", scales="free")
        + theme_minimal()
        + theme(
            axis_text_x=element_text(rotation=45, hjust=1, size=14),
            axis_title=element_blank(),
        )
        + coord_flip()
    )

    return plot


plot = plot_coef_with_ci(X, Y, 10.0)
plot.save("results/regression_coefs.pdf", dpi=300)
