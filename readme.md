# Football Player Goals Prediction - ML Project

## Project objective
Predict a football player's season goal count (`Goals`) from the 2022-2023 Football Player Stats dataset using regression models.

## Dataset
- File: `data/raw/football_player_stats.csv`
- Original format: semicolon-separated CSV (`;`), Latin-1 encoding
- Rows/columns: 2,689 / 124
- Target: `Goals`
- Missing value: one missing value in `Nation`, handled as `Unknown`

## Project workflow (same lifecycle-oriented structure as the Churn Prediction reference)
1. Data validation and exploratory summary
2. Preprocessing and feature engineering
3. Train/test split (80/20, random_state=42)
4. Train and compare regression models
5. Evaluate with MAE, RMSE, R² and 5-fold cross-validation
6. Save model, predictions, plots and metadata
7. Validate data, outputs and reproducibility
8. Optional MLflow tracking/model registry

## Important leakage prevention
`Rk` and `Player` are identifiers and are removed. `G/Sh` and `G/SoT` are goal-derived ratios, so they are removed to avoid giving the model information calculated from the target. The target `Goals` is kept separate from input features.

## Setup (Windows / VS Code)
```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the complete project
From this project root:
```bash
python pipelines/run_full_pipeline.py
```

Or run each stage:
```bash
python src/validate_data.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/validate_outputs.py
python src/validate_reproducibility.py
```

## Optional MLflow
```bash
python pipelines/run_mlflow.py
mlflow ui
```
Open the local MLflow UI address printed by MLflow in your browser.

## Main outputs
- `data/processed/processed_football.csv` - encoded feature matrix and target
- `models/best_goals_model.pkl` - trained regression model
- `artifacts/split_data.json` - reproducible split information
- `artifacts/preprocessing_metadata.json` - preprocessing details
- `outputs/model_comparison.csv` - model performance comparison
- `outputs/regression_metrics.json` - best model metrics
- `outputs/predictions.csv` - actual and predicted goals
- `outputs/actual_vs_predicted.png` - regression plot
- `outputs/residual_plot.png` - residual diagnostic
- `outputs/feature_importance.csv` - feature importance where supported
