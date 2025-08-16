# Hybrid KNN + LSTM — Sequential / Time-Series Modeling

This repository implements a **hybrid** approach for sequence modeling:
1) compute **KNN-derived features** (e.g., KNN regression predictions, mean neighbor distance),  
2) create supervised **windows** (look_back → horizon),  
3) train **LSTM baseline** and **Hybrid (LSTM + KNN features)**, then compare metrics.

Notebook: `notebooks/knn_lstm.ipynb`.

## Why Hybrid?
Neighbor-based signals from KNN can capture local patterns not easily learned by a small LSTM on limited data. We add those signals as extra channels/features to the LSTM input.


## Pipeline
- **Prepare:** time-aware split, scaling (MinMax/Standard).
- **KNN features:** `k` nearest neighbors over selected feature space; export `knn_pred`, `knn_mean_dist` (and others if needed).
- **Windowing:** convert time series to supervised windows (`look_back`, `horizon`).
- **Modeling:** LSTM baseline vs Hybrid (append KNN features to each time step or as exogenous features).
- **Evaluation:** RMSE/MAE/MAPE/R² (regression) or Accuracy/F1 (classification).

## Results (update from notebook)
- **Baseline LSTM:** RMSE = `...`, MAE = `...`, R² = `...`
- **Hybrid KNN + LSTM:** RMSE = `...`, MAE = `...`, R² = `...`
- Plots are stored in `results/plots/` (residuals, ŷ vs y, learning curves).
