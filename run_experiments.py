import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from src.main import train, infer
from src.features import preprocess_features, get_feature_names
from src.model import NTargetModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import os

TARGETS = ["target_w", "target_r", "target_g", "target_b"]

def load_and_split_data():
    """Carica e divide i dati per la validazione."""
    print("--- Caricamento Dati ---")
    X_train = pd.read_parquet("data/X_train.parquet")
    y_train = pd.read_parquet("data/y_train.parquet")
    
    # Split Train/Val
    moons = X_train["moon"].unique()
    split_point = int(len(moons) * 0.8)
    train_moons = moons[:split_point]
    val_moons = moons[split_point:]
    
    X_t = X_train[X_train["moon"].isin(train_moons)]
    y_t = y_train[y_train["moon"].isin(train_moons)]
    X_v = X_train[X_train["moon"].isin(val_moons)]
    y_v = y_train[y_train["moon"].isin(val_moons)]
    
    return X_t, y_t, X_v, y_v

def evaluate_model(X_train, y_train, X_val, y_val, model_name, model_class, model_params):
    """Valuta un modello e restituisce la correlazione media."""
    print(f"\n{'='*60}")
    print(f"Esperimento: {model_name}")
    print(f"{'='*60}")
    
    # Preprocessing
    feature_names = get_feature_names(X_train)
    X_t_processed = preprocess_features(X_train, feature_names)
    X_v_processed = preprocess_features(X_val, feature_names)
    
    # Training
    model = NTargetModel(target_names=TARGETS)
    model.models = {}
    
    for target in TARGETS:
        if target in y_train.columns:
            print(f"Training {model_name} for {target}...")
            m = model_class(**model_params)
            valid_mask = y_train[target].notna()
            if valid_mask.sum() > 0:
                m.fit(X_t_processed[valid_mask], y_train.loc[valid_mask, target])
                model.models[target] = m
    
    # Predizione
    predictions = pd.DataFrame(index=X_v_processed.index)
    for target, m in model.models.items():
        predictions[target] = m.predict(X_v_processed)
    preds = predictions.mean(axis=1)
    
    # Valutazione
    time_col = "date" if "date" in X_val.columns else "moon"
    score_df = pd.DataFrame({
        time_col: X_val[time_col].values,
        "id": X_val["id"].values,
        "prediction": preds
    })
    score_df = pd.merge(score_df, y_val, on=["id", time_col])
    
    correlations = []
    for moon, group in score_df.groupby(time_col):
        corr = spearmanr(group["prediction"], group["target_b"]).correlation
        if not np.isnan(corr):
            correlations.append(corr)
    
    avg_corr = np.mean(correlations)
    print(f"Mean Spearman Correlation: {avg_corr:.6f}")
    
    return avg_corr, model

def run_all_experiments():
    """Esegue tutti gli esperimenti e confronta i risultati."""
    X_t, y_t, X_v, y_v = load_and_split_data()
    
    experiments = [
        {
            "name": "1. LinearRegression (Baseline)",
            "class": LinearRegression,
            "params": {}
        },
        {
            "name": "2. Ridge (alpha=1.0)",
            "class": Ridge,
            "params": {"alpha": 1.0, "random_state": 42}
        },
        {
            "name": "3. Ridge (alpha=10.0)",
            "class": Ridge,
            "params": {"alpha": 10.0, "random_state": 42}
        },
        {
            "name": "4. Lasso (alpha=0.01)",
            "class": Lasso,
            "params": {"alpha": 0.01, "random_state": 42, "max_iter": 2000}
        },
        {
            "name": "5. Random Forest (n=50)",
            "class": RandomForestRegressor,
            "params": {"n_estimators": 50, "max_depth": 5, "random_state": 42, "n_jobs": -1}
        },
        {
            "name": "6. Random Forest (n=100)",
            "class": RandomForestRegressor,
            "params": {"n_estimators": 100, "max_depth": 7, "random_state": 42, "n_jobs": -1}
        },
        {
            "name": "7. XGBoost (depth=3)",
            "class": xgb.XGBRegressor,
            "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, 
                      "objective": "reg:squarederror", "random_state": 42, "n_jobs": -1}
        },
        {
            "name": "8. XGBoost (depth=5)",
            "class": xgb.XGBRegressor,
            "params": {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05, 
                      "objective": "reg:squarederror", "random_state": 42, "n_jobs": -1}
        },
        {
            "name": "9. LightGBM (leaves=31)",
            "class": lgb.LGBMRegressor,
            "params": {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.1, 
                      "random_state": 42, "n_jobs": -1, "verbose": -1}
        },
        {
            "name": "10. LightGBM (leaves=50)",
            "class": lgb.LGBMRegressor,
            "params": {"n_estimators": 150, "num_leaves": 50, "learning_rate": 0.05, 
                      "random_state": 42, "n_jobs": -1, "verbose": -1}
        }
    ]
    
    results = []
    best_score = -np.inf
    best_model_info = None
    
    for exp in experiments:
        try:
            score, model = evaluate_model(
                X_t, y_t, X_v, y_v,
                exp["name"], exp["class"], exp["params"]
            )
            results.append({
                "name": exp["name"],
                "score": score,
                "model": model
            })
            
            if score > best_score:
                best_score = score
                best_model_info = {
                    "name": exp["name"],
                    "score": score,
                    "model": model,
                    "class": exp["class"],
                    "params": exp["params"]
                }
        except Exception as e:
            print(f"Errore durante {exp['name']}: {e}")
            results.append({
                "name": exp["name"],
                "score": None,
                "model": None
            })
    
    # Stampa riepilogo finale
    print(f"\n{'='*60}")
    print("RIEPILOGO FINALE")
    print(f"{'='*60}")
    for r in results:
        if r["score"] is not None:
            print(f"{r['name']:<40} {r['score']:.6f}")
        else:
            print(f"{r['name']:<40} ERRORE")
    
    print(f"\n{'='*60}")
    print("MIGLIOR MODELLO")
    print(f"{'='*60}")
    print(f"Nome: {best_model_info['name']}")
    print(f"Score: {best_model_info['score']:.6f}")
    print(f"Parametri: {best_model_info['params']}")
    
    # Salva il miglior modello
    os.makedirs("resources_best", exist_ok=True)
    best_model_info["model"].save("resources_best")
    print(f"\nMiglior modello salvato in: resources_best/")
    
    return results, best_model_info

if __name__ == "__main__":
    results, best_model = run_all_experiments()
