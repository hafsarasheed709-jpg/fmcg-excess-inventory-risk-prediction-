# FMCG Excess Inventory Risk Prediction

Predicts, every Monday, which products of an FMCG distributor (Multan example) will carry excess stock 7 days later.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard: calendar date pickers, filters, KPI cards, Predict Risk, trends, model performance, what-if, data explorer |
| `generate_fmcg_inventory_data.py` | Creates the simulated dataset (40 products, weekly, 2025-2026) |
| `fmcg_inventory_dataset.csv` | The dataset |
| `train_inventory_model.py` | Trains and compares models, saves `inventory_model.pkl` |
| `inventory_model.pkl` | Trained model (re-create with the training script on your own computer) |
| `data_analysis.py` | EDA report and plots into `analysis_output/` |
| `analysis_output/` | Reports, tables and plots (the dashboard reads two CSV files from here) |
| `presentation/` | 25-slide deck (PPTX and PDF) |

## Run
```
pip install -r requirements.txt
python train_inventory_model.py
streamlit run app.py
```

## Note
Data is simulated. Validate with real distributor records before practical use.
