# SuperKart Sales Forecast - MLOps Pipeline

Automated pipeline that validates the SuperKart dataset, prepares it, tunes tree-ensemble models, commits the best model to the repo and serves it through a Streamlit app.

## Structure
```
app.py                          Streamlit app (loads model/best_model.joblib)
requirements.txt                dependencies for CI and Streamlit Community Cloud
data/SuperKart.csv              raw dataset
src/register_data.py            validate columns and register the dataset
src/data_prep.py                clean, split, save data/processed/train.csv and test.csv
src/train.py                    tune Random Forest and Gradient Boosting, log trials, save best model
model/                          best_model.joblib and metrics.json (committed by the workflow)
experiments/                    tuning_trials.csv and run_summary.json (committed by the workflow)
.github/workflows/pipeline.yml  GitHub Actions: register -> prep -> train -> smoke test
```

## Run locally
```
pip install -r requirements.txt
python src/register_data.py
python src/data_prep.py
python src/train.py
streamlit run app.py
```

## CI/CD
Every push to `main` runs the workflow. The train job commits the new model back to `main` (with `[skip ci]`), and Streamlit Community Cloud redeploys from `main`.

## Results (held-out test set)
Random Forest: R2 0.932, RMSE about 278, MAE about 106.
