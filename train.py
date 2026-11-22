"""Step 3 - Model building with experiment tracking.

Loads the train/test splits (downloaded from the workflow artifact in CI, or
produced locally by data_prep.py), tunes two tree-ensemble models with
GridSearchCV, logs every parameter combination tried, evaluates on the held-out
test set and saves the best model to model/best_model.joblib so the pipeline
can commit it to the repository.

Tracking: every run is always written to experiments/ (CSV + JSON) so the log
is committed with the repo. If MLflow is installed it is used as well.

Usage (from the repo root):
    python src/train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:  # optional - not required for the pipeline to work
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("model")
EXP_DIR = Path("experiments")
TARGET = "Product_Store_Sales_Total"
RANDOM_STATE = 42

NUMERIC = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
CATEGORICAL = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Family",
]

# Models and the parameter grids to tune
CANDIDATES = {
    "random_forest": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [6, 10, 14],
            "model__min_samples_leaf": [1, 3, 5],
        },
    ),
    "gradient_boosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        },
    ),
}


def build_pipeline(model) -> Pipeline:
    """Preprocessing + model in one object, so the app can feed raw columns."""
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
    }


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    print(f"Train {X_train.shape}, Test {X_test.shape}")

    EXP_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    if mlflow is not None:
        mlflow.set_tracking_uri(f"file:{EXP_DIR.resolve() / 'mlruns'}")
        mlflow.set_experiment("superkart-sales-forecast")

    all_trials = []
    summary = {}
    best_name, best_search, best_r2 = None, None, -np.inf

    for name, (model, grid) in CANDIDATES.items():
        print(f"\nTuning {name} ...")
        search = GridSearchCV(
            build_pipeline(model), grid, cv=3, scoring="r2", n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)

        # Log every parameter combination that was tried
        cv = pd.DataFrame(search.cv_results_)
        for _, row in cv.iterrows():
            trial = {"model": name, **{k.replace("model__", ""): v for k, v in row["params"].items()}}
            trial["cv_r2_mean"] = float(row["mean_test_score"])
            trial["cv_r2_std"] = float(row["std_test_score"])
            all_trials.append(trial)

        test_metrics = evaluate(search.best_estimator_, X_test, y_test)
        train_metrics = evaluate(search.best_estimator_, X_train, y_train)
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        summary[name] = {
            "best_params": best_params,
            "cv_r2": float(search.best_score_),
            "train": train_metrics,
            "test": test_metrics,
        }
        print(f"  best params : {best_params}")
        print(f"  CV R2       : {search.best_score_:.4f}")
        print(f"  Test        : {test_metrics}")

        if mlflow is not None:
            with mlflow.start_run(run_name=name):
                mlflow.log_params(best_params)
                mlflow.log_metric("cv_r2", float(search.best_score_))
                for split, m in (("train", train_metrics), ("test", test_metrics)):
                    for k, v in m.items():
                        mlflow.log_metric(f"{split}_{k}", v)

        if test_metrics["r2"] > best_r2:
            best_name, best_search, best_r2 = name, search, test_metrics["r2"]

    # Save logs, metrics and the best model
    pd.DataFrame(all_trials).to_csv(EXP_DIR / "tuning_trials.csv", index=False)
    (EXP_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))

    # compress=3 keeps the file well under GitHub's 25 MB web-upload limit
    joblib.dump(best_search.best_estimator_, MODEL_DIR / "best_model.joblib", compress=3)
    final = {"best_model": best_name, **summary[best_name]}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(final, indent=2))

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name}")
    print(f"Test R2={final['test']['r2']:.4f}  RMSE={final['test']['rmse']:.2f}  MAE={final['test']['mae']:.2f}")
    print(f"Saved model -> {MODEL_DIR / 'best_model.joblib'}")
    print(f"Saved tuning log ({len(all_trials)} trials) -> {EXP_DIR / 'tuning_trials.csv'}")


if __name__ == "__main__":
    main()
