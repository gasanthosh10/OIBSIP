# Task 5 — Sales Prediction Using Python

A regression project that predicts product sales from TV, radio, and newspaper
advertising budgets.

## Method

- Cleans the supplied advertising dataset
- Uses a fixed holdout test set for honest evaluation
- Compares linear regression and random forest
- Selects the model with the highest test R²
- Reports R², MAE, and RMSE
- Provides an interactive advertising-budget simulator

## Run

```bash
python train_model.py
streamlit run app.py
```

## Interpretation

The dashboard exposes channel influence and observed relationships. Predictions
represent model estimates and do not prove that advertising alone caused sales.

